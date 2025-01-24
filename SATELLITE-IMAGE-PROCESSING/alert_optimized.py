import pandas as pd
from geopy.distance import geodesic
import folium
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def process_spill_data(spill_data_path, ais_data_path, ports_data_path,save_dir_path):
    
    # Load and preprocess data
    spill_data = pd.read_csv(spill_data_path)
    ais_data = pd.read_csv(ais_data_path)
    ports_data = pd.read_csv(ports_data_path, encoding='latin1')

    spill_data['BaseDateTime'] = pd.to_datetime(spill_data['BaseDateTime'])
    ais_data['BaseDateTime'] = pd.to_datetime(ais_data['BaseDateTime'])
    ports = ports_data[['LAT', 'LON', 'Port_Name', 'Email']].to_dict('records')

    PROXIMITY_THRESHOLD = 100  # kilometers
    TIME_WINDOW = 3600  # seconds (1 hour)
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT = 587
    SENDER_EMAIL = "ismailsudais2005@gmail.com"
    SENDER_PASSWORD = "zlzz ywtm ycdk qftq"

    def calculate_distance(lat1, lon1, lat2, lon2):
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers

    def find_nearby_ships(spill, ais_data, seen_ships):
        nearby_ships = []
        for _, row in ais_data.iterrows():
            if row['MMSI'] != spill['MMSI'] and abs((row['BaseDateTime'] - spill['BaseDateTime']).total_seconds()) <= TIME_WINDOW:
                distance = calculate_distance(spill['LAT_x'], spill['LON_x'], row['LAT'], row['LON'])
                if distance <= PROXIMITY_THRESHOLD and row['MMSI'] not in seen_ships:
                    nearby_ships.append({
                        'MMSI': row['MMSI'],
                        'LAT': row['LAT'],
                        'LON': row['LON'],
                        'BaseDateTime': row['BaseDateTime'],
                        'VesselName': row.get('VesselName', 'Unknown'),
                        'distance': distance
                    })
                    seen_ships.add(row['MMSI'])
        return pd.DataFrame(nearby_ships)

    def find_nearest_port(spill, ports):
        min_distance = float('inf')
        nearest_port = None
        for port in ports:
            distance = calculate_distance(spill['LAT_x'], spill['LON_x'], port['LAT'], port['LON'])
            if distance < min_distance:
                min_distance = distance
                nearest_port = {**port, 'Distance': distance}
        return nearest_port

    def create_map(spill, nearby_ships, nearest_port):
        m = folium.Map(location=[spill['LAT_x'], spill['LON_x']], zoom_start=10)
        folium.Marker(
            [spill['LAT_x'], spill['LON_x']],
            popup=f'Spill Ship: {spill.get("VesselName_x", "Unknown")} (MMSI: {spill["MMSI"]})',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        for _, ship in nearby_ships.iterrows():
            folium.Marker(
                [ship['LAT'], ship['LON']],
                popup=f'{ship["VesselName"]} (MMSI: {ship["MMSI"]})\nDistance: {ship["distance"]:.2f} km',
                icon=folium.Icon(color='blue', icon='ship')
            ).add_to(m)
        if nearest_port:
            folium.Marker(
                [nearest_port['LAT'], nearest_port['LON']],
                popup=f'Nearest Port: {nearest_port["Port_Name"]}\nDistance: {nearest_port["Distance"]:.2f} km',
                icon=folium.Icon(color='green', icon='anchor')
            ).add_to(m)

        map_file = os.path.join(save_dir_path, f'map_spill_ship_{spill["MMSI"]}.html')
        m.save(map_file)
        print(map_file)
        return map_file

    def send_alert_email(spill, nearby_ships, nearest_port, map_file):
        if not nearest_port:
            print("No nearby port found for alert.")
            return
        subject = f"Oil Spill Alert: {spill.get('VesselName_x', 'Unknown')} (MMSI: {spill['MMSI']})"
        body = f"""
        <html>
            <body>
                <p><strong>{nearest_port['Port_Name']} Authorities</strong>,</p>
                <p>An oil spill has been detected. Below are the details:</p>
                <ul>
                    <li><strong>Spill Ship:</strong> {spill.get('VesselName_x', 'Unknown')} (MMSI: {spill['MMSI']})</li>
                    <li><strong>Location:</strong> Latitude {spill['LAT_x']}, Longitude {spill['LON_x']}</li>
                    <li><strong>Nearby Ships:</strong></li>
                    {nearby_ships.to_html(index=False)}
                </ul>
                <p>Respond promptly to mitigate environmental damage.</p>
            </body>
        </html>
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = nearest_port['Email']
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))
            with SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
            print(f"Alert sent to {nearest_port['Port_Name']} ({nearest_port['Email']}).")
        except Exception as e:
            print(f"Failed to send alert: {e}")

    # Main processing loop
    for _, spill in spill_data.iterrows():
        seen_ships = set()  # Reset seen ships for each spill
        nearby_ships = find_nearby_ships(spill, ais_data, seen_ships)
        nearest_port = find_nearest_port(spill, ports)
        map_file = create_map(spill, nearby_ships, nearest_port)
        send_alert_email(spill, nearby_ships, nearest_port, map_file)

    print("All alerts processed.")

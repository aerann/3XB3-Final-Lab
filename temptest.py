import csv
from math import radians, sin, cos, sqrt, atan2

class LondonSubwayGraph:
    def __init__(self, stations_file, connections_file):
        self.graph = self.build_graph(stations_file, connections_file)

    def build_graph(self, stations_file, connections_file):
        graph = {}
        # Read station data (latitude, longitude) from csv
        with open(stations_file, 'r') as stations_csv:
            stations_reader = csv.reader(stations_csv)
            next(stations_reader)  # Skip header
            for row in stations_reader:
                station_id, latitude, longitude = map(int, row)
                graph[station_id] = {'latitude': latitude, 'longitude': longitude, 'neighbors': {}}

        # Read connections data from csv
        with open(connections_file, 'r') as connections_csv:
            connections_reader = csv.reader(connections_csv)
            next(connections_reader)  # Skip header
            for row in connections_reader:
                station1, station2 = map(int, row)
                distance = self.calculate_distance(graph[station1], graph[station2])
                graph[station1]['neighbors'][station2] = distance
                graph[station2]['neighbors'][station1] = distance

        return graph

    def calculate_distance(self, station1, station2):
        # Function to calculate Euclidean distance between two stations based on latitude and longitude
        lat1, lon1 = radians(station1['latitude']), radians(station1['longitude'])
        lat2, lon2 = radians(station2['latitude']), radians(station2['longitude'])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Radius of Earth in kilometers (approximate)
        earth_radius = 6371.0

        # Calculate distance
        distance = earth_radius * c
        return distance



# Example usage
stations_file_path = 'london_stations.csv'
connections_file_path = 'london_connections.csv'
london_subway = LondonSubwayGraph(stations_file_path, connections_file_path)

# Access the graph (dictionary where keys are station IDs and values are dictionaries with latitude, longitude, and neighbors)
graph_data = london_subway.graph
print(graph_data)
```
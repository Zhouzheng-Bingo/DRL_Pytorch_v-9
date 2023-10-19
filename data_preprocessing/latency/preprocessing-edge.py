import re
import csv


# Function to extract all required data, including throughput, from the file
def extract_all_data_with_throughput_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

        # Extract all data preprocessing latencies
        data_preprocessing_pattern = r"Data preprocessing latencies:\s*\[(.*?)\]"
        data_preprocessing_matches = re.findall(data_preprocessing_pattern, content, re.DOTALL)

        # Extract all Layer-wise latencies for ResNet
        resnet_pattern = r"Layer-wise latencies for ResNet \(repeated\):\s*\[(.*?)\]"
        resnet_data_matches = re.findall(resnet_pattern, content, re.DOTALL)

        # Extract all Layer-wise latencies for TCN
        tcn_pattern = r"Layer-wise latencies for TCN \(repeated\):\s*\[(.*?)\]"
        tcn_data_matches = re.findall(tcn_pattern, content, re.DOTALL)

        # Extract all Data transmission latencies
        transmission_pattern = r"Data transmission latencies \(edge\):\s*\[(.*?)\]"
        transmission_data_matches = re.findall(transmission_pattern, content, re.DOTALL)

        # Extract all Throughput data
        throughput_pattern = r"Throughput:\s*\[(.*?)\]"
        throughput_matches = re.findall(throughput_pattern, content, re.DOTALL)

        return data_preprocessing_matches, resnet_data_matches, tcn_data_matches, transmission_data_matches, throughput_matches


# Convert string data to list of floats
def convert_to_float_list(data_str, delimiter=','):
    return [float(item.strip()) for item in data_str.split(delimiter)]


# Extract all data with throughput from server file
if __name__ == '__main__':

    # Extract all data with throughput from edge file
    edge_all_data_preprocessing, edge_all_resnet_data, edge_all_tcn_data, edge_all_transmission_data, edge_all_throughput = extract_all_data_with_throughput_from_file(
        "./data/results-edge.txt")

    # Process and save edge data for all groups to CSV
    edge_csv_data_all_groups = [
        ["Group", "Task Number", "Latency (Edge)", "Data Transmission Latency (Edge)", "Throughput (Edge)"]]

    # Process each group and append to the csv data list
    for group_num in range(len(edge_all_data_preprocessing)):
        edge_data_preprocessing_list = convert_to_float_list(edge_all_data_preprocessing[group_num])
        edge_resnet_data_list = convert_to_float_list(edge_all_resnet_data[group_num])
        edge_tcn_data_list = convert_to_float_list(edge_all_tcn_data[group_num])
        edge_transmission_data_list = convert_to_float_list(edge_all_transmission_data[group_num], delimiter=' ')
        edge_throughput_list = convert_to_float_list(edge_all_throughput[group_num])

        # Merge all edge data into a single list for this group
        edge_latencies = edge_data_preprocessing_list + edge_resnet_data_list + edge_tcn_data_list

        for i, latency in enumerate(edge_latencies, start=1):
            edge_csv_data_all_groups.append(
                [group_num + 1, f"Task {i}", latency, edge_transmission_data_list[i - 1], edge_throughput_list[i - 1]])

    # Save to CSV
    with open("./data/edge_data_all_groups.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(edge_csv_data_all_groups)

    # You can repeat similar steps for the server file

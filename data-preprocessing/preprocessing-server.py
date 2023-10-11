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

    # Extract all data with throughput from server file
    server_all_data_preprocessing, server_all_resnet_data, server_all_tcn_data, _, server_all_throughput = extract_all_data_with_throughput_from_file(
        "./data/results-server.txt")

    # Process and save server data for all groups to CSV
    server_csv_data_all_groups = [["Group", "Task Number", "Latency (Server)", "Throughput (Server)"]]

    # Process each group and append to the csv data list
    for group_num in range(len(server_all_data_preprocessing)):
        server_data_preprocessing_list = convert_to_float_list(server_all_data_preprocessing[group_num])
        server_resnet_data_list = convert_to_float_list(server_all_resnet_data[group_num])
        server_tcn_data_list = convert_to_float_list(server_all_tcn_data[group_num])
        server_throughput_list = convert_to_float_list(server_all_throughput[group_num])

        # Merge all server data into a single list for this group
        server_latencies = server_data_preprocessing_list + server_resnet_data_list + server_tcn_data_list

        for i, latency in enumerate(server_latencies, start=1):
            server_csv_data_all_groups.append([group_num + 1, f"Task {i}", latency, server_throughput_list[i - 1]])

    # Save to CSV
    with open("./data/server_data_all_groups.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(server_csv_data_all_groups)

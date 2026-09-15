# Import
import numpy as np
import csv
import math

from typing import List, Dict

# 1. Sensor readings
def read_sensor_data(filepath: str) -> Dict[List[float], int]:
    """
        Read PM2.5 sensor reading into a list.

        Parameter
        ---------
        filepath
            str: path directory of the csv file.
        
        Returns
        -------
        dict{list[float], int}
            A dictionary with list of pm2.5 readings and count.
    """
    # Read the csv file
    with open(filepath, "r") as f:
        csv_reader = csv.DictReader(f, delimiter = ";")

        # Iterate
        sensor_data = []
        for r in csv_reader:
             if r["value_type"] == "P2":
                sensor_data.append(float(r["value"]))

    return {
        "data" : sensor_data, 
        "count": len(sensor_data), 
        "type": type(sensor_data), 
        "d_type": type(sensor_data[0])}

# 2. Compute average
def compute_average(sensor_values: List[float]) -> float:
    """
        Compute the average of sensor readings.

        Parameters
        ----------
        sensor_values: List[float]
            A list containing float pm2.5 readings.
        
        Returns
        ------
            float
                Mean value for pm2.5 values.
    """
    total = 0.0

    for value in sensor_values:
        total += value
    
    return total / len(sensor_values)

# 3. Compute above/below average
def above_below_average(sensor_values: List[float], above_avg: bool = True) -> Dict[List, int]:
    """
        Return a list of values that are above/below mean.

        Parameters
        ----------
        sensor_value: List[float]
            A list of float sensor readings.

        above_avg: bool
            A boolean selector to determine choice, either get values above average or below average. By default, above average sensor values are returned.

        Returns
        -------
        Dict[List, int]
            A dictionary containing filtered values (above or below average) and COUNT of those values. 
    """
    # Get mean
    mean = compute_average(sensor_values)

    # Value above 
    above_below_avg = []
    if above_avg:
        # Loop
        for sensor_value in sensor_values:
            if sensor_value > mean:
                above_below_avg.append(sensor_value)

    # Value below mean
    else:
        for sensor_value in sensor_values:
            if sensor_value < mean:
                above_below_avg.append(sensor_value)

    # Return
    return {"data": above_below_avg, "count": len(above_below_avg)}
# 4. Compute std 
def compute_std(sensor_values: List[float])-> float:
    """
        Compute standard deviation using Bessel's correction(n - 1) method.

        Parameters
        ----------
        sensor_values: List[float]
            A list of sensor readings.

        Returns
        -------
        Float
            Standard deviation.
    """
    # Get mean
    mean = compute_average(sensor_values)

    # Compute Squared difference
    avg_sqr_diff = 0.0
    for sensor_value in sensor_values:
        sqr_diff = ((sensor_value - mean) ** 2)
        avg_sqr_diff += sqr_diff

    # Compute variance
    var = avg_sqr_diff / (len(sensor_values) - 1)

    # Return std
    return math.sqrt(var)

# 5. Readings within 1-std of the mean
def within_1_std(sensor_values: List[float]) -> Dict[List[list, int], List[list, int]]:
    """
        Lists values that are within one standard deviaton, above and below average(i.e mean).

        Parameters
        ----------
        sensor_data: List[float]
            A list of sensor data, containing float entries

        Returns
        ------
        Dict[List[list, int], List[list, int]]
            A dictionary with two lists. One with positive values i.e above one standard deviation and the other with negative values i.e with values below one standard deviation. 

            Each list have another list, containing filtered sensor values and an integer with total number of items present in that list 
    """
    # Get mean
    mean = compute_average(sensor_values)

    # Get standard deviation
    std = compute_std(sensor_values)

    # Lower and upper bound
    lower_bound = mean - std
    upper_bound = mean + std

    # List
    below_1_std = []
    above_1_std = []

    # Loop
    for sensor_value in sensor_values:
        if (sensor_value >= lower_bound) & (sensor_value <= upper_bound):
            if sensor_value < mean:
                below_1_std.append(sensor_value)
            else:
                above_1_std.append(sensor_value)


    # Return 
    return {"above_1_std": [above_1_std, len(above_1_std)], 
    "below_1_std": [below_1_std, len(below_1_std)]}


# 6. Compute min-max
def min_max(sensor_values: List[float]) -> Dict[float, float]:
    """
        Get the minimum and maximum sensor values.

        Parameters
        ----------
        sensor_values: List[float]
            A list with sensor values (i.e PM2.5) reading.

        Returns
        -------
        Dict[float, float]
            A dictionary with two floats, one with maximum reading and the other with minimum reading.
    """
    # Instantanciate
    max = sensor_values[0]
    min = max

    # Loop
    for sensor_value in sensor_values:
        # Get maximum sensor value
        if sensor_value > max:
            max = sensor_value

        # Get minimum sensor value
        if sensor_value < min:
            min = sensor_value

    # Return
    return {"maximum": max, "minimum": min}

# Summary function
def summary(filename: str, data_filepath: str, sensor_values: List[float]) -> Dict[list[bool]]:
    """
        A summary statistics highlighting: mean, standard deviation, maximum, minimum, values above and below one standard deviaton, and a sample data used for analysis.

        Parameters
        ----------
        filename: str
            Filename to save the summary statistics.
        sensor_values: List[float]
            A list containing floats with sensor readings.
        data_filepath: str
            A string path, where data is located.(PWD for data/)

        Returns
        -------
        Dict[list[bool]]
            A dictionary with lists, each list with a boolean True or False, indicating success summary statistic for a particular operation.
    """
    # Get statistics
    # 1. Mean
    mean = compute_average(sensor_values)

    # 2. Standard deviation
    std = compute_std(sensor_values)

    # 3. Maximum
    min_max_dict = min_max(sensor_values) 
    max = min_max_dict["maximum"]

    # 4. Minimum
    min = min_max_dict["minimum"]

    # 5. Some values below one standard deviation
    values_within_1_std = within_1_std(sensor_values)
    values_below, below_count = values_within_1_std["below_1_std"]

    # 6. Some values above one standard deviation
    values_above, above_count = values_within_1_std["above_1_std"]

    # 7. Some values above average
    above_avg = above_below_average(sensor_values)
    abv_data = above_avg["data"]
    abv_count = above_avg["count"]

    # 8. Some values below average
    below_avg = above_below_average(sensor_values, False)
    blw_data = below_avg["data"]
    blw_count = below_avg["count"]

    # Open, write and close text file
    with open(filename, "w") as f:
        # Header
        f.write("Summary data analysis for PM2.5 values\n")
        # 5 sensor data
        if read_sensor_data:
            f.write("\nSample dataset: \n")
            [f.write(f"{str(sensor)}\t") for sensor in sensor_values[:5]]
            f.write(f"\n\nLength of the dataset: {len(sensor_values)}")
            write_data_success = True

        # Mean
        if mean:
            f.write(f"\n\nMean value is: {mean:3f}\n")
            write_mean_success = True

        # Standard deviation
        if std:
            f.write(f"\nStandard deviation is: {std:3f}\n")
            write_std_success = True

        # Check for a standard normal distribution
        is_std_normal_dist = np.isclose(mean, 1) & np.isclose(std, 0)
        if is_std_normal_dist:
            f.write("\nPM2.5 values follows a standard normal distribution\n")
        if not is_std_normal_dist:
            f.write(f"\nPM2.5 values does not follow a standard normal distribution.\n")
        
        # Minimum sensor reading
        if min:
            f.write(f"\nMinimum reading: {min}\n")
            write_min_success = True

        # Maximum reading
        if max:
            f.write(f"\nMaximum reading: {max}\n")
            write_max_success = True

        # Sample values below one standard deviation
        if values_below:
            f.write(f"\nThere are {blw_count} values below one standard deviation, here is a sample: \n")
            [f.write(f"{str(value)}\t") for value in values_below[:5]]
            f.write("\n")
            write_values_below_success = True

        if values_above:
            f.write(f"\nThere are {abv_count} values above one standard deviation, here is a sample: \n")
            [f.write(f"{str(value)}\t") for value in values_above[:5]]
            f.write("\n")
            write_values_above_success = True

        # Values above average
        if above_avg:
            f.write(f"\nThere are {abv_count} values that are above average. Here is a sample: \n")
            [f.write(f"{str(data)}\t") for data in abv_data[:5]]
            f.write("\n")
            write_above_avg_success = True

        # Values below average
        if below_avg:
            f.write(f"\nThere are {blw_count} values that are below average. Here is a sample: \n")
            [f.write(f"{str(data)}\t") for data in blw_data[:5]]
            f.write("\n")
            write_below_avg_success = True


    # Return
    return {
        "below_avg_success": write_below_avg_success,
        "above_avg_success": write_below_avg_success,
        "values_above_success": write_values_above_success,
        "values_below_success": write_values_below_success,
        "max_success": write_max_success,
        "min_success": write_min_success,
        "std_success": write_std_success,
        "mean_success": write_mean_success,
        "data_success": write_data_success
    }
            
# Main function
def main():
    filename = "summary.txt"
    data_filepath = "data/nairobi_sensor.csv"
    read_sensors = read_sensor_data(data_filepath)
    sensor_values = read_sensors["data"]
    summary_dict = summary(filename, data_filepath, sensor_values)
    print(summary_dict)
# Run
if __name__ == "__main__":
    main()
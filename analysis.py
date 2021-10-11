#!/usr/bin/env python
# coding: utf-8

import sys
import datetime
import math
from tkinter.tix import S_REGION
from typing import List, Tuple
import numpy as np
from numpy.lib.function_base import corrcoef
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import figure
import matplotlib.dates as mdates
import pprint as pp
# import requests
import gc
from influxdb import InfluxDBClient
from calendar import monthrange

from pandas.core.frame import DataFrame
pd.options.mode.chained_assignment = None  # default='warn'
import ipdb # for debugging
# ipdb.set_trace()
pd.set_option('display.max_rows', 100)
from numpy.random import seed
from scipy.stats import pearsonr


# Example: Loads data from the "Power1" circuit from a file downloaded from the PhiSaver website.
# df = pd.read_csv("data/steve-mar-2021.csv", sep=';', index_col=0, dtype=object)
# df = df.sort_values(by='Time', ascending=True)
# df = df.iloc[1:,:]
# powerpoints_1=df.loc[:, "Power1"]
    
# Removes invalid data and converts everything to whole number-rounded ints
def process_row(series: pd.Series) -> pd.Series:
    series1 = pd.Series([0 if str(row).startswith("'") else row for row in series])
    series1 = series1.apply(lambda row: int(round(float(row),0)) if not math.isnan(row) else 0)
    series1.index = series.index
    return series1

# Returns a tuple of shape (closest_index, closest_value)
# which contains the closest value to the given value in the given array
def get_nearest(arr: List[int], value: int) -> Tuple[int, int]:
    closest = sys.maxsize
    closest_index = sys.maxsize
    for index, i in enumerate(arr):
        if (abs(value-i) < abs(value-closest)):
            closest = i
            closest_index = index
    return closest_index, closest

# Returns an ordered series where the indexes are power values and the first values are the frequencies.
# The first element in the series is the most frequent value in the dataset
# Calling this a second time (with the result of the first call passed in) groups the frequencies more tightly
def get_frequencies(freqs: pd.Series, series: pd.Series) -> pd.Series:
    new_values = []
    new_freqs = []
    i = 1
    j = 0

    # initialise lists to contain first item in the value_counts()
    new_values.append(freqs.iloc[0:0+1].index.values[0])
    new_freqs.append(freqs[freqs.iloc[0:0+1].index.values[0]])

    for value, freq in freqs.iteritems():
        # last item so checking 'next' item will throw IndexError
        if (value == series.value_counts().iloc[-1:].index.values[0]):
            i+=1
            j+=1
            continue

        current_value = freqs.iloc[j:j+1].index.values[0]
        compared_index, compared_value = get_nearest(new_values, current_value)

        # basically the same
        if (abs(current_value - compared_value) < 8 ):
            new_values[compared_index] = (current_value + compared_value) / 2
            new_freqs[compared_index] = freqs[current_value] + new_freqs[compared_index]

        # not the same so insert new entry 
        else:
            new_values.append(current_value)
            new_freqs.append(freqs[current_value])
            i+=1
        j+=1    

    return pd.Series(new_freqs, index=new_values).sort_values(ascending=False)

# Gets the two most frequent power readings of the data.
# From this, we can determine the cycling power values of the refrigerator.
def get_modes(data: pd.Series) -> Tuple[int, int]:
    
    # First mode
    value_counts = data.value_counts()
    ser = value_counts.index
    mode = ser[0]
    count = 0
    for i in ser:
        if (abs(i-mode) < 10):
            count += 1
        else:
            break

    val_total = 0
    freq_total = 0
    index = 0
    while (index < count):
        val_total += value_counts.iloc[index] * ser[index]
        freq_total += value_counts.iloc[index]
        index += 1
    mode = int(round((val_total / freq_total), 2))
    
    # Second mode
    i = 0
    for index, _ in value_counts.iteritems():
        if (not (abs(ser[i] - mode) < 10)):
            second_mode = ser[i]
            break
        i+=1

    return mode, second_mode

# Returns True if the peak value (less the peak start value) is within the range specified (less the peak start value)
def isInRange(peak_value: int, trough_value: int, min_power: int, max_power: int) -> bool:
    if ((peak_value - trough_value >= min_power) and
        (peak_value - trough_value <= max_power)):
        return True
    return False

# Returns a series of tuples of shape: (peak_start_time, peak_max_value, peak_duration),
# where each peak is within the range specified, finishes within the specified duration, and 
# is off for at least off_requirement minutes after the peak
def get_peaks(data: pd.Series, min_power: int, max_power: int, min_required_duration: int = 0, max_required_duration: int = sys.maxsize, off_requirement: int = 0, granularity: int = 1/12) -> List[Tuple[str, int, int]]:
    i = 1
    peaks = []
    while i < len(data):
        peak_duration = granularity # +1 tick for the initial turning on
        try:
            curr_idx = i
            prev_idx = i-1
            next_idx = i+1
            climb_start_idx = 0
            peak_idx = 0
            # if granularity == 1 and i > 2000 and i < 2050:
            #     print(data.index[i])
            if (data.iloc[curr_idx] - data.iloc[prev_idx] > 100): # could be the start of a peak
                climb_start_idx = prev_idx # mark start of climb...
                peak_idx = curr_idx
                while(True):
                    curr_off_duration = off_requirement
                    if (data.iloc[next_idx] - data.iloc[curr_idx] > 0): # still rising... not at peak yet
                        curr_idx+=1
                        prev_idx+=1
                        next_idx+=1
                        if (data.iloc[curr_idx] > data.iloc[peak_idx]):
                            peak_idx=curr_idx
                        peak_duration += granularity
                    elif (abs(data.iloc[next_idx] - data.iloc[curr_idx]) < 100): # not rising but still at the same peak
                        curr_idx+=1
                        prev_idx+=1
                        next_idx+=1
                        peak_duration += granularity
                    elif (data.iloc[next_idx] > data.iloc[climb_start_idx] + 150): # not at (close enough to) being a trough yet
                        curr_idx+=1
                        prev_idx+=1
                        next_idx+=1
                        peak_duration += granularity
                    else: # at a trough, so append climb_start which is the initial peak
                        if (peak_duration <= max_required_duration and 
                            isInRange(data.iloc[peak_idx], data.iloc[climb_start_idx], min_power, max_power)):
                            satisfied_off_requirement = True
                            while (curr_off_duration > 0):
                                if (data.iloc[next_idx+1] > data.iloc[next_idx] + 100): # hasn't been a trough for long enough
                                    satisfied_off_requirement = False
                                    break
                                curr_off_duration -= granularity
                                curr_idx+=1
                                prev_idx+=1
                                next_idx+=1
                            if (not satisfied_off_requirement):
                                curr_idx = next_idx
                                prev_idx = curr_idx-1
                                next_idx+=1
                                continue
                            if (peak_duration >= min_required_duration):
                                peaks.append((data.index[peak_idx], data.iloc[peak_idx] - data.iloc[climb_start_idx], round(peak_duration)))
                        break
            i = next_idx
        except IndexError:
            i = next_idx
    
    return peaks

# Example usages of getPeaks()
# food_prep_peaks = get_peaks(powerpoints_1, 700, 2200, 0, 10, 0)
# microwave_peaks = get_peaks(powerpoints_1, 600, 1200, 0, 10, 0)
# stove_top_peaks = get_peaks(stove, 500, 5000, 0, 120, 60)

# Returns True if the value is between the start and end times
def timeInRange(value: str, start: str, end: str, days_overlap: bool) -> bool:
    midnight_mins = 1440
    
    # get all strings into int's representing total minutes
    value_mins = (int(value[0:2]) * 60) + int(value[3:5])
    start_mins = (int(start[0:2]) * 60) + int(start[3:5])
    end_mins = (int(end[0:2]) * 60) + int(end[3:5])
     
    # if the end time is in the early hours of the next day, alter the value time and end time by adding 24hours  
    if (days_overlap):
        end_mins += 24 * 60
        if not (midnight_mins - value_mins < midnight_mins - start_mins):
            value_mins += (24 * 60)
    
    if (value_mins <= end_mins and value_mins >= start_mins):
        return True
    else:
        return False

# Returns the average time in the given data as a formatted string (hh:mm:ss)
def get_average_wake_or_sleep(data: List[Tuple[str, str, int]], overnight_cases: bool = False) -> str:
    total_mins = 0
    for row in data:
        row_mins = (int(row[0][11:13]) * 60) + int(row[0][14:16])
        
        if (overnight_cases and row_mins <= 120):
            row_mins += (24 * 60)
        total_mins += row_mins
    
    hours = (total_mins / (len(data) * 60)) % 24
    hours = int(math.floor(hours))
    mins = total_mins % 60
     
    dt = datetime.datetime(2000, 1, 1, hours, mins, 0)
    
    return dt.strftime("%X")

# Returns the most common sleep time and wake time (i.e. circadian rhythm)
# TODO: Ignore weekends for a more accurate result
def get_circadian_rhythm(data: pd.Series) -> Tuple[str, str]:
    i = 0
    on_cases = []
    off_cases = []
    resting_power = get_modes(data)[0]
    while i < len(data):
        try:
            on_duration = 0
            off_duration = 0

            while (data.iloc[i] > resting_power): # lights are on
                on_duration += 1
                i+=1
            
            if on_duration > 120 and timeInRange(data.index[i][11:], '19:00:00', '02:00:00', True): # lights have been on for more than 120 minutes
                off_cases.append((data.index[i], 'turning off', on_duration))

            while (data.iloc[i] <= resting_power): # lights are off
                off_duration += 1
                i+=1
                
            if off_duration > 120 and timeInRange(data.index[i][11:], '03:00:00', '10:00:00', False): # lights have been off for more than 120 minutes
                on_cases.append((data.index[i], 'turning on', off_duration))

            i+=1
        
        except IndexError:
            i+=1
            
    average_wake = get_average_wake_or_sleep(on_cases)
    average_sleep = get_average_wake_or_sleep(off_cases, True)
    
    return (average_wake, average_sleep)

# Returns the most common left the house and got home from work (i.e. work schedule)
# TODO: Ignore weekends for a more accurate result
def get_work_schedule(data: pd.Series) -> Tuple[str, str]:
    i = 0
    on_cases = []
    off_cases = []
    resting_power = get_modes(data)[0]
    while i < len(data):
        try:
            on_duration = 0
            off_duration = 0

            while (data.iloc[i] > resting_power): # lights are on
                on_duration += 1
                i+=1
            
            if on_duration > 120 and timeInRange(data.index[i][11:], '05:00:00', '10:00:00', False): # lights have been on for more than 120 minutes
                off_cases.append((data.index[i], 'turning off', on_duration))

            while (data.iloc[i] <= resting_power): # lights are off
                off_duration += 1
                i+=1
                
            if off_duration > 120 and timeInRange(data.index[i][11:], '14:00:00', '19:00:00', False): # lights have been off for more than 120 minutes
                on_cases.append((data.index[i], 'turning on', off_duration))

            i+=1
        
        except IndexError:
            i+=1
            
    average_home = get_average_wake_or_sleep(on_cases)
    average_leave = get_average_wake_or_sleep(off_cases, True)
    
    return (average_leave, average_home)

# Returns the energy usage (Wh) of each peak
def get_watt_hours(peaks: List[Tuple[str,int,int]], data: pd.Series) -> List[Tuple[str,int]]:
    watt_hours = []
    for peak in peaks:
        watt_hour = 0
        for _ in range(peak[2]):
            watt_hour += data.loc[peak[0]]
        watt_hours.append((peak[0], watt_hour))

    return watt_hours

# Returns a list of daily Wh values with each entry representing one day of the month.
def get_daily_usage(arr: List[Tuple[str,int]]) -> pd.Series:
    df = pd.DataFrame(arr).set_index(0)[1]
    df.index.name = 'Time'
    df.name = 'Processed Watt Hours'
    df.index = pd.to_datetime(df.index)
    df = df.resample('D').sum()
    
    start_date = int(df.index[0].strftime("%d"))
    i = start_date - 1
    while (i > 0):
        datetime = pd.Timestamp(int(df.index[0].strftime("%Y")), int(df.index[0].strftime("%m")), i, 0, 0, 0)
        df.loc[datetime] = 0
        df.sort_index(inplace=True) 
        i-=1
    
    end_date = int(df.index[-1].strftime("%d"))
    i = end_date + 1
    days_in_month = monthrange(int(df.index[-1].strftime("%Y")), int(df.index[-1].strftime("%m")))[1]
    while (i <= days_in_month):
        datetime = pd.Timestamp(int(df.index[-1].strftime("%Y")), int(df.index[-1].strftime("%m")), i, 0, 0, 0)
        df.loc[datetime] = 0
        df.sort_index(inplace=True) 
        i+=1
        
    return df

# Loads and processes data from the BOM. E.g. http://www.bom.gov.au/climate/dwo/202101/html/IDCJDW4019.202101.shtml
def load_bom_data(filename: str) -> List[int]:
    all_data = pd.read_csv(filename, sep=',', header=None, dtype=object)
    data = []
    for i in range(6):
        month = all_data.iloc[:, i]
        month = month.dropna()
        data.append([int(float(month.iloc[j])) if month.iloc[j] != "NaN" else True for j in range(len(month))])
    return data

# Example usage
# mar_2021 = load_bom_data("data/extra/bom-mar-2021.csv", True)
# dec_2020 = load_bom_data("data/extra/bom-dec-2020.csv", True)
# jan_2021 = load_bom_data("data/extra/bom-jan-2021.csv", True)
# feb_2021 = load_bom_data("data/extra/bom-feb-2021.csv", True)
# max_temps_summer = dec_2020 + jan_2021 + feb_2021

# jun_2020 = load_bom_data("data/extra/bom-jun-2020.csv", False)
# jul_2020 = load_bom_data("data/extra/bom-jul-2020.csv", False)
# aug_2020 = load_bom_data("data/extra/bom-aug-2020.csv", False)
# min_temps_winter = jun_2020 + jul_2020 + aug_2020

# aircon_peaks_winter = get_peaks(aircon_winter, 100, 10000, 0, 1000, 60)
# aircon_peaks_summer = get_peaks(aircon_summer, 100, 10000, 0, 1000, 60)

# watt_hours_winter = get_watt_hours(aircon_peaks_winter, aircon_winter)
# watt_hours_summer = get_watt_hours(aircon_peaks_summer, aircon_summer)

# usage_series_winter = get_daily_usage(watt_hours_winter)
# usage_series_summer = get_daily_usage(watt_hours_summer)

# Plots the Wh aircon usage against the temperatures of the month (from BOM above)
# TODO: Consider using another metric for heat sensitivity
def plot_aircon_trend(usage_series: pd.Series, temps: List[int], s_type: str):
    usage = usage_series.tolist()
    x_ticks = [i for i in range(len(usage)) if i % 4 == 0]
    x_ticklabels = [usage_series.index[i].strftime("%Y/%m/%d") for i in x_ticks]

    n=len(usage)
    position = np.arange(n)
    offset = 0.15
    width = 0.3

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax.bar(position - offset, temps, width, label = s_type + " Temperature", color='orange')
    ax2.bar(position + offset, usage, width, label = 'Aircon Usage', color='blue')
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    
    min_temp = min(temps)
    max_temp = max(temps)

    ax2.set_ylabel("Power W")
    ax.set_ylabel("Degree's C")
    ax.set_ylim([min_temp-2,max_temp+2])

    ax.set_xlabel("Date")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, rotation=60)
    
    plt.title(s_type + " Temperature vs. Aircon Usage")
    plt.show()

# plot_aircon_trend(usage_series, mar_2021, s_type="Max")
# plot_aircon_trend(usage_series_winter, min_temps_winter, s_type="Min")
# plot_aircon_trend(usage_series_summer, max_temps_summer, s_type="Max")

# Returns the probability of the aircon being used given a temperature using Bayes' Theorem.
def get_probabilities(temps: List[int], usage_series: pd.Series):
    usage = usage_series.tolist()
    
    td = pd.DataFrame(temps)
    td = [int(float(td.iloc[i])) for i in range(len(td))]
    td = pd.Series(td)

    n = pd.DataFrame()
    n['temps'] = td
    n['vals'] = usage
    
    num_days = len(temps)
    
    probs = []
    min_temp = min(temps)
    max_temp = max(temps)
    
    # Bayes' Theorem
    for i in range(min_temp, max_temp+1):
        p_a = usage_series.gt(0).sum() / num_days
        p_b = td.eq(i).sum() / num_days
        p_a_or_b = (n['vals'].gt(0) | n['temps'].eq(i)).sum() / num_days
        p_a_and_b = p_a + p_b - p_a_or_b
        if (p_b == 0):
            probs.append(np.nan)
            continue
        p_a_given_b = p_a_and_b / p_b
        probs.append(round(p_a_given_b,2))
    
    return probs
        
# get_probabilities(mar_2021, usage_series)

# Plots the probability curve from above for all temperatures of the month.
def plot_probability_curve(temps: List[int], usage_series: pd.Series):
        
    min_temp = min(temps)
    max_temp = max(temps)
        
    plt.figure(figsize=(8,6))
        
    probs = get_probabilities(temps, usage_series)
    interpolated = pd.Series(probs).interpolate().values.ravel().tolist()
    
    y = probs
    y = np.array(y)
    x = np.arange(min_temp, max_temp+1, 1)
        
    plt.plot(x, interpolated)
    
    ax = plt.gca()
    ax.set_ylim([0.0,1.0])
    
    plt.xlabel("Temperature (C)")
    plt.ylabel("Probability of Aircon Use")
    plt.title("Probability Curve of Aircon Usage vs Temperature")
    plt.show()
    
    return x, interpolated

# plot_probability_curve(mar_2021, usage_series)
# plot_probability_curve(max_temps_summer, usage_series_summer)
# plot_probability_curve(min_temps_winter, usage_series_winter)

# Joins the summer and winter probability calculations and displays them on the same graph.
# TODO: Maybe filter this so that only major aircon usages are plotted.
def plot_probs_combined(min_temps_winter: List[int], usage_series_winter: pd.Series, max_temps_summer: List[int], usage_series_summer: pd.Series):
    winter_probs = get_probabilities(min_temps_winter, usage_series_winter)
    summer_probs = get_probabilities(max_temps_summer, usage_series_summer)
    
    all_temps = np.arange(min(min_temps_winter), max(max_temps_summer)+1, 1)
    all_probs = [0.0] * len(all_temps)
    
    i=0
    while (i < len(winter_probs)):
        all_probs[i] = winter_probs[i]
        i+=1

    j = len(all_probs) - len(summer_probs)
    while (j < len(all_probs)):
        all_probs[j] = summer_probs[j-len(summer_probs)-1]
        j+=1
        
    interpolated = pd.Series(all_probs).interpolate().values.ravel().tolist()
    
    plt.figure(figsize=(8,6))
    plt.plot(all_temps, interpolated)

    ax = plt.gca()
    ax.set_ylim([0.0,1.0])

    plt.xlabel("Temperature (C)")
    plt.ylabel("Probability of Aircon Use")
    plt.title("Probability Curve of Aircon Usage vs Temperature")
    plt.show()
    
    return all_temps, all_probs

# plot_probs_combined(min_temps_winter, usage_series_winter, max_temps_summer, usage_series_summer)


# ------------ DATABASE INTERACTIONS START HERE ------------ #

# client = InfluxDBClient(host='live2.phisaver.com', database='phisaver', username='reader', password='Rmagine!', port=8086, headers={'Accept': 'application/json'}, gzip=True)

# These kinds of structures are hard-coded dictionaries which tell us the specific circuit name for each category (i.e. oven, main powerpoint, lights, aircon). 
oven_circuits = {   
                    "uq10": "Oven",
                    "uq12": "none",
                    "uq23": "none",
                    "uq24": "OvenHob",
                    "uq26": "none",
                    "uq33": "none",
                    "uq37": "Oven",
                    "uq45": "Oven",
                    "uq48": "WallOven",
                    "uq49": "Oven",
                    "uq56": "Oven",
                    "uq57": "Oven",
                    "uq61": "Oven",
                    "uq67": "none", 
                    "uq68": "Oven",
                    "uq75": "Oven",
                    "uq85": "HotPlateOven",
                    "uq88": "Oven",
                    "uq92": "none",
                    "hfs01a": "Oven",
                }

# TODO: Change this to be the circuit with the most kitchen-like activity (not neccessarily the fridge's circuit)
power_circuits = {                     
                    "uq10": "Powerpoints1",
                    # "uq12": "Powerpoints2",
                    # "uq23": "Powerpoints1",
                    # "uq24": "Powerpoints2",
                    # "uq26": "Powerpoints1",  
                    # "uq33": "Powerpoints2",
                    # "uq37": "PowerpointsRear",
                    # "uq45": "Powerpoints1",
                    # "uq48": "Powerpoints2",
                    # "uq49": "Powerpoints1",
                    # "uq56": "Powerpoints2",
                    # "uq57": "Powerpoints4",
                    # "uq61": "Misc1",
                    # "uq67": "Powerpoints1", 
                    # "uq68": "Powerpoints1",
                    # "uq75": "Powerpoints1",
                    # "uq85": "Powerpoints2",
                    # "uq88": "Powerpoints1",
                    # "uq92": "Powerpoints1",
                    "hfs01a": "Power1",
                }

stove_circuits = {  
                    "uq10": "Oven",
                    "uq12": "Stove",
                    "uq23": "Stove",
                    "uq24": "OvenHob",
                    "uq26": "Stove",
                    "uq33": "Stove",
                    "uq37": "Hotplate",
                    "uq45": "Oven",
                    "uq48": "Hotplates",
                    "uq49": "Hob",
                    "uq56": "Oven",
                    "uq57": "Oven",
                    "uq61": "Oven",
                    "uq67": "Stove", 
                    "uq68": "Hotplate",
                    "uq75": "Hob",
                    "uq85": "HotPlateOven",
                    "uq88": "Oven",
                    "uq92": "Stove",
                    "hfs01a": "Hotplate"
                }


# Transforms a UTC time string into a timezone aware time string
def localTimeFromUtc(start, end):
    start = (datetime.datetime.strptime(start, "'%Y-%m-%dT%H:%M:%SZ'") - datetime.timedelta(hours=10)).strftime("'%Y-%m-%dT%H:%M:%SZ'")
    end = (datetime.datetime.strptime(end, "'%Y-%m-%dT%H:%M:%SZ'") - datetime.timedelta(hours=10)).strftime("'%Y-%m-%dT%H:%M:%SZ'")
    return start, end

# Returns a DateFrame containing a bunch of other DataFrames of raw data from the PhiSaver InfluxDB.
# The chunked=True parameter to the InfluxDB client is important. This ensures that the server is not overloaded 
# and only retrieves a certain amount of data with any one call.
# Inputs: The parameters to call the query with.
def get_data_in_period(client, start_date, end_date, device, sensor):
    # Change from UTC to local time
    start_date, end_date = localTimeFromUtc(start_date, end_date)
    current_date = start_date
    dataF = pd.DataFrame()
    
    while (current_date < end_date):
        q = """SELECT * FROM "iotawatt" WHERE "device" = {} AND "sensor" = {} AND "time" >= {} AND "time" <= {}""".format(device, sensor, current_date, end_date)
        print(q)
        data = pd.DataFrame(client.query(q, chunked=True))
        if (data.empty):
            return dataF
        dataF = dataF.append(data, ignore_index=True)
        most_recent_time = get_most_recent_time(dataF)
        if (current_date == most_recent_time or most_recent_time == -1):
            break
        current_date = most_recent_time
    return dataF

# Gets the string representing the time of the last retrieved entry from the DB.
def get_most_recent_time(df):
    last_idx = df.index[-1]
    last_arr = df[0][last_idx]
    if (last_arr is None):
        return -1
    last_time = last_arr[-1].get("time")
    return "'{}'".format(last_time)

# Apply a certain level of granularity over the data
# Default was 5sec == 1/12min == 0.0833min
def apply_granularity(df: pd.DataFrame, granularity: int) -> pd.DataFrame:
    default_granularity = 1/12  # 5sec (for hfs01a) or 10sec (for others e.g. uqXX)
    if (granularity == 1/12):
        df = df.resample('5S').max()
    elif (granularity == 1/6):
        df = df.resample('10S').max()
    else:
        df = df.resample('{}min'.format(int(round(12 * granularity * default_granularity)))).max()
    return df

# Example usage of iteratively calling the InfluxDB
# microwave_dataframes = []
# for i, home in enumerate(power_circuits):
#     microwave_dataframes.append(get_data_in_period(client, "'2021-03-01T00:00:00Z'", "'2021-03-31T23:59:50Z'", "'{}'".format(home), "'{}'".format(power_circuits[home])))

# After retrieving bulk data from the db, call this function.
# It will call the process_row() function and return a list of DF's
def pre_process(dataframes, granularity = 1/12): #granularity in mins = 1/12 mins = 0.0833 mins = 5 seconds = default granularity of db
    post_process = []
    if (not isinstance(dataframes, list)):
        new_df = []
        new_df.append(dataframes)
        dataframes = new_df
    
    for dataframe in dataframes:
        df = pd.DataFrame()
        if (dataframe.empty):
            post_process.append(df)
            continue
        for inner_df in dataframe[0]:
            to_append = pd.DataFrame(inner_df)
            df = df.append(to_append)

        df = df.set_index('time')
        df = df.set_index(pd.to_datetime(df.index))
        df = df.loc[:, "Watts"]
        df = df.to_frame()
        # if (granularity >= 1/12):
            # k = int(round(12 * granularity))
            # idx = df.index[::k]
        df = apply_granularity(df, granularity)
            # if (df.size != len(idx)):
            #     df = df.drop(df.tail(df.size - len(idx)).index)
            
        df.index = [(datetime.datetime.strptime(x.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=10)).strftime('%Y-%m-%dT%H:%M:%SZ') for x in df.index]

            # to_append = to_append.set_index('time')
            # to_append = to_append.set_index(pd.to_datetime(to_append.index))
            # to_append = to_append.loc[:, "Watts"]
            # to_append = to_append.to_frame()
            # if (granularity >= 1):
            #     k = int(round(12 * granularity))
            #     idx = to_append.index[::k]
            #     print('INDEX: ', idx)
            #     to_append = apply_granularity(to_append, granularity)
            #     if (to_append.size != len(idx)):
            #         to_append = to_append.drop(to_append.tail(to_append.size - len(idx)).index)
                
            # to_append.index = [(datetime.datetime.strptime(x.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=10)).strftime('%Y-%m-%dT%H:%M:%SZ') for x in to_append.index]
            # pp.pprint(to_append)
            
            # df = df.append(to_append)
        
        processing = process_row(df.squeeze())

        post_process.append(processing)
        
    for df in post_process:
        df.index = df.index.to_series().apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S'))

    return post_process

# Gets the microwave and stove peaks
def microwave_stove_peaks(processed, stove_processed, granularity):
    microwave = []
    stove = []
    
    for df in processed:
        microwave_peaks = get_peaks(df, 600, 1200, 1, 10, 0, granularity)
        pp.pprint(microwave_peaks)
        microwave.append(len(microwave_peaks))
        
    for df in stove_processed:
        stove_peaks = get_peaks(df, 500, 5000, 1, 120, 60, granularity)
        pp.pprint(stove_peaks)
        stove.append(len(stove_peaks))
        
    return microwave, stove

# microwave_peaks, stove_peaks = microwave_stove_peaks(processed, stove_processed)

# Normalisation function.
def normalise(x, x_min, x_max):
    try:
        return round(2*((x-x_min)/(x_max-x_min))-1,2)
    except ZeroDivisionError:
        return 0

# ---------- Played around with using http requests to post data to a web app at one point ---------- #

# # data={'microwave': hfs01a_microwave_normalised, 'stovetop': hfs01a_stove_normalised}
# data={'microwave': uq49_microwave_normalised, 'stovetop': uq49_stove_normalised}

# r = requests.post('http://localhost:3001/data', data=data)
# print(r.json())
# print(r.status_code)

# ---------------------------------------------------------------------------------------------------- #

aircon_circuits = {
                    "uq10": "none",
                    "uq12": "Aircon1",
                    "uq23": "none",
                    "uq24": "Aircon2",
                    "uq26": "Aircon",
                    "uq33": "Aircon2",
                    "uq37": "Aircon1",
                    "uq45": "Aircon",
                    "uq48": "AirconShed",
                    "uq49": "Aircon1",
                    "uq56": "Aircon",
                    "uq57": "Aircon",
                    "uq61": "Aircon2",
                    "uq67": "Aircon", 
                    "uq68": "Aircon",
                    "uq75": "Aircon1",
                    "uq85": "Aircon1OP",
                    "uq88": "Aircon1",
                    "uq92": "Lights1",
                    "hfs01a": "Aircon1"
                }


light_circuits = {
                    "uq10": "Lights1",
                    "uq12": "Lights12",
                    "uq23": "Lights1",
                    "uq24": "Lights1",
                    "uq26": "Light",
                    "uq33": "Lights1",
                    "uq37": "Lights12",
                    "uq45": "Lights",
                    "uq48": "Lights1",
                    "uq49": "Lights1",
                    "uq56": "Lights1",
                    "uq57": "Lights2",
                    "uq61": "Lights1",
                    "uq67": "Lights1", 
                    "uq68": "Lights",
                    "uq75": "Lights",
                    "uq85": "Lights3",
                    "uq88": "Lights1",
                    "uq92": "Lights1",
                    "hfs01a": "Lights2",
                }

# Gets all the circadian rhythm's.
def get_all_circadian_rhythm(light_processed):
    lst = []
    for i in range(len(light_processed)):
        try:
            lst.append(get_circadian_rhythm(light_processed[i]))
        except ZeroDivisionError:
            print("zero div err")
            lst.append(0)
    return lst

# Deprecated:
# Return the average circadian rhythm's and nicely put them into a JSON object to be sent via http.
def get_average_circadian_rhythm_time(times):
    total_morning_seconds = 0
    total_night_seconds = 0
    min_morning = '23:59:59'
    max_morning = '00:00:01'
    min_night = '23:59:59'
    max_night = '00:00:01'
    
    fmt = "%H:%M:%S"
    
    for tup in times:
        total_morning_seconds += int(tup[0][6:8])
        total_morning_seconds += int(tup[0][3:5]) * 60
        total_morning_seconds += int(tup[0][0:2]) * 60 * 60
        
        total_night_seconds += int(tup[1][6:8])
        total_night_seconds += int(tup[1][3:5]) * 60
        total_night_seconds += int(tup[1][0:2]) * 60 * 60
        
        if (datetime.datetime.strptime(tup[0], fmt) < datetime.datetime.strptime(min_morning, fmt)):
            min_morning = tup[0]
        if (datetime.datetime.strptime(tup[0], fmt) > datetime.datetime.strptime(max_morning, fmt)):
            max_morning = tup[0]
            
        if (datetime.datetime.strptime(tup[1], fmt) < datetime.datetime.strptime(min_night, fmt)):
            min_night = tup[1]
        if (datetime.datetime.strptime(tup[1], fmt) > datetime.datetime.strptime(max_night, fmt)):
            max_night = tup[1]
        
    seconds = total_morning_seconds
    hours, seconds =  str((seconds // 3600) // len(times)).zfill(2), seconds % 3600
    minutes, seconds = str(seconds // 60).zfill(2), str(seconds % 60).zfill(2)
    average_morning = "{}:{}:{}".format(hours, minutes, seconds)
    
    seconds = total_night_seconds
    hours, seconds =  str((seconds // 3600) // len(times)).zfill(2), seconds % 3600
    minutes, seconds = str(seconds // 60).zfill(2), str(seconds % 60).zfill(2)
    average_night = "{}:{}:{}".format(hours, minutes, seconds)
    
    return {
            'min_morning': min_morning,
            'average_morning': average_morning,
            'max_morning': max_morning,
            'min_night': min_night,
            'average_night': average_night,
            'max_night': max_night
           }

# ---------- More http request stuff ---------- #

# microwave_stovetop = {'microwave': uq49_microwave_normalised, 'stovetop': uq49_stove_normalised}
# circadian_rhythm_metrics = circadian_rhythm_data
# circadian_rhythm_test = {'morning': '05:28:00', 'night': '22:21:00'}
# data={'microwave_stovetop': microwave_stovetop, \
#       'circadian_rhythm_metrics': circadian_rhythm_metrics, \
#       'circadian_rhythm_test': circadian_rhythm_test, \
#      }

# r = requests.post('http://localhost:3001/data', json=data)
# print(r.json())
# print(r.status_code, '\n')
# pp.pprint(data)

# --------------------------------------------- #

# Returns -1 if a < b. Returns 1 if a > b. Returns 0 if a == b.
# TODO: Add a 2 hour buffer around the circadian rhythm time
def compareTime(first, second):
    a = datetime.datetime.strptime(first, '%H:%M:%S')
    b = datetime.datetime.strptime(second, '%H:%M:%S')
    if (a < b):
        return -1
    elif (a > b):
        return 1
    else:
        return 0

# Get spikes (Watts greater than houses resting power use)
# of less than 30 mins in duration between their circadian rhythm times.
def get_sleep_disturbances(data):
    sleep_disturbances = []
    for household in data:
        curr_disturbances = get_peaks(household, 5, 1000, 1, 30, 60)
        try:
            circadian_rhythm = get_circadian_rhythm(household)
            curr_disturbances = [peak for peak in curr_disturbances if compareTime(peak[0][11:], circadian_rhythm[0]) < 0 or compareTime(peak[0][11:], circadian_rhythm[1]) > 0]
            sleep_disturbances.append(curr_disturbances)
        except ZeroDivisionError:
            sleep_disturbances.append(-1)

    return [len(arr) if type(arr) != int else arr for arr in sleep_disturbances]

# Gets the work schedules for all households.
def get_work_schedules(data):
    lst = []
    for household in data:
        try:
            lst.append(get_work_schedule(household))
        except ZeroDivisionError:
            lst.append(0)
    return lst

# Return the number of hours between the two dates.
def hoursBetween(first, second, overnight=False):
    a = datetime.datetime.strptime(first, '%H:%M:%S')
    b = datetime.datetime.strptime(second, '%H:%M:%S')
    if (overnight == True):
        temp = b
        b = a + datetime.timedelta(days=1)
        a = temp
    diff = b - a
    return int(diff.total_seconds() // 60)

# Gets the work durations for all households.
def get_work_durations(work_schedules):
    duration = []
    for tup in work_schedules:
        if (tup == 0):
            duration.append(-2)
        else:
            duration.append(hoursBetween(tup[0], tup[1]))
    return duration        

# Gets the sleep durations for all households.
def get_sleep_durations(sleep_schedules):
    duration = []
    for tup in sleep_schedules:
        if (tup == 0):
            duration.append(-2)
        else:
            duration.append(hoursBetween(tup[0], tup[1], True))
    return duration  

# The maximum age of the occupants in each household.
ages = { 
        "uq10": 40,
        "uq12": 38,
        "uq23": 49,
        "uq24": 53,
        "uq26": 41,
        "uq33": 44,
        "uq37": 65,
        "uq45": 40,
        "uq48": 55,
        "uq49": 27,
        "uq56": 52,
        "uq57": 65,
        "uq61": 56,
        "uq67": 47, 
        "uq68": 50,
        "uq75": 30,
        "uq85": 38,
        "uq88": 55,
        "uq92": 33,
        "hfs01a": 35,
       }

# Value is 1 if the house has a specific stove circuit, 0 otherwise (indicating they have a gas stove).
stove = { 
        "uq10": 0,
        "uq12": 1,
        "uq23": 1,
        "uq24": 1,
        "uq26": 1,
        "uq33": 1,
        "uq37": 1,
        "uq45": 0,
        "uq48": 1,
        "uq49": 1,
        "uq56": 0,
        "uq57": 0,
        "uq61": 0,
        "uq67": 1, 
        "uq68": 1,
        "uq75": 1,
        "uq85": 1,
        "uq88": 0,
        "uq92": 1,
        "hfs01a": 1,
       }

# Value is 1 if the house has a specific hotwater circuit, 0 otherwise (indicating they have gas hotwater).
hotwater = { 
        "uq10": 1,
        "uq12": 1,
        "uq23": 0,
        "uq24": 1,
        "uq26": 1,
        "uq33": 1,
        "uq37": 1,
        "uq45": 0,
        "uq48": 0,
        "uq49": 0,
        "uq56": 1,
        "uq57": 0,
        "uq61": 0,
        "uq67": 0, 
        "uq68": 1,
        "uq75": 1,
        "uq85": 1,
        "uq88": 1,
        "uq92": 0,
        "hfs01a": 1,
       }


# Takes in the normalised values of all the variables of measuring stress/anxiety and returns a rank to be displayed.
def stress_anxiety(sleep_disturbances, work_durations, sleep_durations, microwave, stovetop, ages, consumptions):
    print("Sleep Disturbances: \t", sleep_disturbances)
    print("Work Durations: \t", work_durations)
    print("Sleep Durations: \t", sleep_durations)
    print("Microwave: \t\t", microwave)
    print("Stovetop: \t\t", stovetop)
    print("Ages: \t\t", ages)
    print("Consumption: \t", consumptions)
    
    ranks = [0] * len(ages)
    for i, sleep_disturbance in enumerate(sleep_disturbances):
        if sleep_disturbance != -2: 
            if (sleep_disturbance > 0.6):
                ranks[i] += 2
            elif (sleep_disturbance > 0.2):
                ranks[i] += 1
    for i, work_duration in enumerate(work_durations):
        if work_duration != -2:
            if (work_duration > 0.6):
                ranks[i] += 2
            elif (work_duration > 0.2):
                ranks[i] += 1
    for i, sleep_duration in enumerate(sleep_durations):
        if sleep_duration != -2:
            if (sleep_duration < 0.6):
                ranks[i] += 2
            elif (sleep_duration < 0.2):
                ranks[i] += 1
    for i in range(len(microwave)):
        if (microwave[i] > 0.4 and stovetop[i] < -0.4):
            ranks[i] += 2
        elif (microwave[i] > 0.2 and stovetop[i] < -0.2):
            ranks[i] += 1
    for i, age in enumerate(ages.values()):
        if age >= 65:
            ranks[i] -= 2
    for i, consumption in enumerate(consumptions):
        if consumption != -2:
            if (consumption < 0.6):
                ranks[i] += 2
            elif (consumption < 0.2):
                ranks[i] += 1
            
    for rank in ranks:
        if (rank < 0):
            rank = 0
    
    print("Stress/Anxiety Ranks: \t\t\t", ranks)
    print()
    return ranks

# Takes in the normalised values of all the variables of measuring home risk and returns a rank to be displayed.
def home_risk(stove, hotwater, ages, consumptions, ovens):
    print()
    print("Stove?: \t\n", stove)
    print("Hotwater?: \t\n", hotwater)
    print("Ages: \t\t\n", ages)
    print("Consumption: \t\n", consumptions)
    print("Oven: \t\t\n", ovens)
    print()

    ranks = [0] * len(stove)
    for i, val in enumerate(stove.values()):
        if val == 0:
            ranks[i] += 1
    for i, val in enumerate(hotwater.values()):
        if val == 0:
            ranks[i] += 1
    for i, age in enumerate(ages.values()):
        if age >= 65:
            ranks[i] += 2
    for i, consumption in enumerate(consumptions):
        if consumption != -2:
            if (consumption < 0.6):
                ranks[i] += 2
            elif (consumption < 0.2):
                ranks[i] += 1
    for i, oven in enumerate(ovens):
        if oven != -2:
            if (oven > 0.6):
                ranks[i] += 2
            elif (oven > 0.2):
                ranks[i] += 1
    
    for rank in ranks:
        if (rank < 0):
            rank = 0
    
    print("Home Risk Ranks: \t\t\t\n", ranks)
    print()
    return ranks

# Returns a ranked value between 0 and 100
def rank(oldValue):
    oldMin = -1
    oldMax = 1 
    newMin = 0
    newMax = 100 # The scale we want to show in the visualisations/output
    
    oldRange = (oldMax - oldMin)  
    newRange = (newMax - newMin)  
    newValue = (((oldValue - oldMin) * newRange) / oldRange) + newMin
    
    return int(newValue)

# Final normalisation and ranking with respect to other households in dataset
# Displays key, value pairs == 'household': rank
def normalise_and_rank(ranks, names):
    values = {}
    for i in range(len(ranks)):
        normalised = normalise(ranks[i], min(ranks), max(ranks))
        ranked = rank(normalised)
        values[names[i]] = ranked
    try:
        pp.pprint(list(values.values()))
    except AttributeError:
        print(list(values.values()))
    return values

# result = normalise_and_rank(stress_anxiety_all, list(light_circuits.keys()))

# The main function for processing the stress/anxiety calculation. 
# Iterating through the months of the sample, it reads from the db, calls the helper functions above, and ranks the households against each other.
def process_stress_anxiety():
    months = ['2020-11-01', '2020-12-01', '2020-12-31']
    
    final_result = {}
    
    for j in range(len(months)-1):
        print("--- Processing {} => {} ---".format(months[j], months[j+1]))
        
        # Microwave Data
        print("Getting microwave data")
        microwave_dataframes = []
        for i, home in enumerate(power_circuits):
            microwave_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(power_circuits[home])))
        
        # Stove Data
        print("Getting stove data")
        stove_dataframes = []
        for i, home in enumerate(stove_circuits):
            stove_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(stove_circuits[home])))           
        
        # Preprocessing 
        print("Processing microwave data")
        microwave_processed = pre_process(microwave_dataframes)
        print("Processing stove data")
        stove_processed = pre_process(stove_dataframes)
        
        
        # 1. Microwave vs Stovetop Use
        # TODO: Stove circuits currently uses a mixture of hotplate and oven circuits. Need to discuss this.
        print("1. Calculating microwave vs stovetop use")
        microwave_peaks, stove_peaks = microwave_stove_peaks(microwave_processed, stove_processed)
        pp.pprint(microwave_peaks)
        pp.pprint(stove_peaks)
        
        microwave_normalised = []
        for val in microwave_peaks:
            microwave_normalised.append(normalise(val, min(microwave_peaks), max(microwave_peaks)))
            
        stovetop_normalised = []
        for val in stove_peaks:
            stovetop_normalised.append(normalise(val, min(stove_peaks), max(stove_peaks)))
           
           
        # Light Data
        print("Getting lights data")
        light_dataframes = []
        for i, home in enumerate(light_circuits):
            light_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(light_circuits[home])))
        
        # Preprocessing
        print("Processing lights data")
        light_processed = pre_process(light_dataframes)
        
        
        # 2. Sleep Disturbance
        print("2. Calculating sleep disturbance")
        sleep_disturbances = get_sleep_disturbances(light_processed)
        pp.pprint(sleep_disturbances)
        
        sleep_disturbances_normalised = []
        for val in sleep_disturbances:
            if val == -1:
                sleep_disturbances_normalised.append(-2)
            else:
                sleep_disturbances_normalised.append(normalise(val, min([x for x in sleep_disturbances if x != -1]), max(sleep_disturbances)))

                
        # 3. Work Duration
        print("3. Calculating work duration")
        work_schedules = get_work_schedules(light_processed)
        
        work_durations = get_work_durations(work_schedules)
        pp.pprint(work_durations)
        
        work_durations_normalised = []
        for val in work_durations:
            if val == -2:
                work_durations_normalised.append(-2)
            else:
                work_durations_normalised.append(normalise(val, min([x for x in work_durations if x != -2]), max(work_durations)))
        
        
        # 4. Sleep Duration
        print("4. Calculating sleep duration")
        sleep_schedules = get_all_circadian_rhythm(light_processed)

        sleep_durations = get_sleep_durations(sleep_schedules)
        pp.pprint(sleep_durations)
        
        sleep_durations_normalised = []
        for val in sleep_durations:
            if val == -2:
                sleep_durations_normalised.append(-2)
            else:
                sleep_durations_normalised.append(normalise(val, min([x for x in sleep_durations if x != -2]), max(sleep_durations)))
        
        # 5. Age
        print("5. Assessing age")
        
        # Consumption Data
        # TODO: Use Consumption circuit instead of power circuit?
        print("Getting consumption data")
        consumption_dataframes = []
        for i, home in enumerate(power_circuits):
            consumption_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(power_circuits[home])))
        
        # Preprocessing
        print("Processing consumption data")
        consumption_processed = pre_process(consumption_dataframes)
        
        # 6. Occupancy
        print("6. Calculating occupancy")
        consumptions = get_all_consumptions(consumption_processed)
        pp.pprint(consumptions)
        
        consumption_normalised = []
        for val in consumptions:
            if val == -2:
                consumption_normalised.append(-2)
            else:
                consumption_normalised.append(normalise(val, min([x for x in consumptions if x != -2]), max(consumptions)))
        
        # Ranking
        print("Ranking results...")
        stress_anxiety_all = stress_anxiety(sleep_disturbances_normalised, work_durations_normalised, sleep_durations_normalised, microwave_normalised, stovetop_normalised, ages, consumption_normalised)
        
        result = normalise_and_rank(stress_anxiety_all, list(light_circuits.keys()))
        
        for key in list(result.keys()):
            try:
                final_result[key] = final_result[key] + result[key]
            except KeyError:
                final_result[key] = result[key]
        print("Current result: ", final_result)
        print()

        del microwave_dataframes
        del stove_dataframes
        del microwave_processed
        del stove_processed
        del light_dataframes
        del light_processed
        del consumption_dataframes
        del consumption_processed
        gc.collect()

    
    for key in list(final_result.keys()):
        final_result[key] = final_result[key] // (len(months)-1)
    
    print("--- Final result ---\n", final_result)        
    return final_result

# res = process_stress_anxiety()

# Gets the total W of power consumption in the data.
def get_consumption(data):
    total = 0
    for val in data:
        total += val
    return int((total / len(data)) * (60*60*24/10))

# Gets all the consumptions for all the households.
def get_all_consumptions(data):
    consumptions = []
    for household in data:
        consumptions.append(get_consumption(household))
    return consumptions

# Gets the oven peaks of > 2 hours
def get_oven(data):
    res = get_peaks(data, 800, 10000, 120, 720, 0)
    if len(res) < 1:
        return -2
    else:
        return len(res)

# Gets the oven peaks for all the households.
def get_all_ovens(data):
    ovens = []
    for household in data:
        if (household.empty):
            ovens.append(-2)
        else:
            ovens.append(get_oven(household))
    return ovens

# The main function for processing the home risk calculation. 
# Iterating through the months of the sample, it reads from the db, calls the helper functions above, and ranks the households against each other.
def process_home_risk():
    months = ['2020-12-01', '2021-01-01'] #, '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2020-12-31']
    
    final_result = {}
    
    for j in range(len(months)-1):
        print("--- Processing {} => {} ---".format(months[j], months[j+1]))
        
        # 1. Gas Stove?
        print("1. Calculating stove infrastructure")
        pp.pprint(stove)
        
        # 2. Gas Hotwater?
        print("2. Calculating hotwater infrastructure")
        pp.pprint(hotwater)
        
        # 3. Age
        print("3. Assessing age")
        pp.pprint(ages)
        
        
        # Consumption Data
        print("Getting consumption data")
        consumption_dataframes = []
        for i, home in enumerate(power_circuits):
            consumption_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(power_circuits[home])))
            
        # Preprocessing
        print("Processing consumption data")
        consumption_processed = pre_process(consumption_dataframes)
        
        # 6. Occupancy
        print("6. Calculating occupancy")
        consumptions = get_all_consumptions(consumption_processed)
        pp.pprint(consumptions)
        
        consumption_normalised = []
        for val in consumptions:
            if val == -2:
                consumption_normalised.append(-2)
            else:
                consumption_normalised.append(normalise(val, min([x for x in consumptions if x != -2]), max(consumptions)))
    
    
        # Oven Data
        print("Getting oven data")
        oven_dataframes = []
        for i, home in enumerate(oven_circuits):
            if (oven_circuits[home] != "none"):
                oven_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(oven_circuits[home])))
            else:
                oven_dataframes.append(pd.DataFrame())
                
        # Preprocessing
        print("Processing oven data")
        oven_processed = pre_process(oven_dataframes)
        
        # 7. Oven on for > 2 hours?
        print("7. Calculating prolonged oven usages")
        ovens = get_all_ovens(oven_processed)
        pp.pprint(ovens)
        
        oven_normalised = []
        for val in ovens:
            if val == -2:
                oven_normalised.append(-2)
            else:
                oven_normalised.append(normalise(val, min([x for x in ovens if x != -2]), max(ovens)))
    
        
        # Ranking
        print("Ranking results...")
        home_risk_all = home_risk(stove, hotwater, ages, consumption_normalised, oven_normalised)
        
        result = normalise_and_rank(home_risk_all, list(stove.keys()))

        print("--- Final result ---") 
        pp.pprint(result)
        
    #     for key in list(result.keys()):
    #         try:
    #             final_result[key] = final_result[key] + result[key]
    #         except KeyError:
    #             final_result[key] = result[key]
    #     # print("Current result: ", final_result)
    #     print()
    
    # for key in list(final_result.keys()):
    #     final_result[key] = final_result[key] // (len(months)-1)
    
    # print("--- Final result ---") 
    # pp.pprint(final_result)       
    # return final_result
    return result


# Takes in the normalised values of all the variables of measuring health insurance and returns a rank to be displayed.
def health_insurance(ages, microwave, stovetop, sleep_disturbances, sleep_durations):
    
    ranks = [0] * len(ages)
    for i, sleep_disturbance in enumerate(sleep_disturbances):
        if sleep_disturbance != -2: 
            if (sleep_disturbance > 0.6):
                ranks[i] += 2
            elif (sleep_disturbance > 0.2):
                ranks[i] += 1
    for i, sleep_duration in enumerate(sleep_durations):
        if sleep_duration != -2:
            if (sleep_duration < 0.6):
                ranks[i] += 2
            elif (sleep_duration < 0.2):
                ranks[i] += 1
    for i in range(len(microwave)):
        if (microwave[i] > 0.4 and stovetop[i] < -0.4):
            ranks[i] += 2
        elif (microwave[i] > 0.2 and stovetop[i] < -0.2):
            ranks[i] += 1
    for i, age in enumerate(ages.values()):
        if age >= 65:
            ranks[i] -= 2
            
    for rank in ranks:
        if (rank < 0):
            rank = 0
    
    print("Health Insurance Ranks: \t\t\t", ranks)
    print()
    return ranks

# The main function for processing the home risk calculation. 
# Iterating through the months of the sample, it reads from the db, calls the helper functions above, and ranks the households against each other.
def process_health_insurance(granularity=1/12): #granularity is in mins
    months = ['2020-07-01', '2020-08-01']
    
    final_result = {}

    client = InfluxDBClient(host='live2.phisaver.com', database='phisaver', username='reader', password='Rmagine!', port=8086, headers={'Accept': 'application/json'}, gzip=True)
    
    for j in range(len(months)-1):
        print("\n\n\n\n\n--- Processing {} => {} ---".format(months[j], months[j+1]))
        
        # 1. Age
        print("1. Assessing age")

        # Microwave Data
        print("Getting microwave data")
        microwave_dataframes = []
        for i, home in enumerate(power_circuits):
            microwave_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(power_circuits[home])))
        
        # Stove Data
        print("Getting stove data")
        stove_dataframes = []
        for i, home in enumerate(stove_circuits):
            stove_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(stove_circuits[home])))           
        
        # Preprocessing 
        print("Processing microwave data")
        microwave_processed = pre_process(microwave_dataframes, granularity)
        print("Processing stove data")
        stove_processed = pre_process(stove_dataframes, granularity)

        del microwave_dataframes
        del stove_dataframes  
        
        # 2. Microwave vs Stovetop Use
        # TODO: Stove circuits currently uses a mixture of hotplate and oven circuits. Need to discuss this.
        print("2. Calculating microwave vs stovetop use")
        microwave_peaks, stove_peaks = microwave_stove_peaks(microwave_processed, stove_processed, granularity)
        pp.pprint(microwave_peaks)
        pp.pprint(stove_peaks)

        del microwave_processed
        del stove_processed
        
        microwave_normalised = []
        for val in microwave_peaks:
            microwave_normalised.append(normalise(val, min(microwave_peaks), max(microwave_peaks)))
            
        stovetop_normalised = []
        for val in stove_peaks:
            stovetop_normalised.append(normalise(val, min(stove_peaks), max(stove_peaks)))
        
        # Light Data
        print("Getting lights data")
        light_dataframes = []
        for i, home in enumerate(light_circuits):
            light_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(light_circuits[home])))

        # Preprocessing
        print("Processing lights data")
        light_processed = pre_process(light_dataframes)

        del light_dataframes

        # 3. Sleep Disturbance
        print("3. Calculating sleep disturbance")
        sleep_disturbances = get_sleep_disturbances(light_processed)
        pp.pprint(sleep_disturbances)
        
        sleep_disturbances_normalised = []
        for val in sleep_disturbances:
            if val == -1:
                sleep_disturbances_normalised.append(-2)
            else:
                sleep_disturbances_normalised.append(normalise(val, min([x for x in sleep_disturbances if x != -1]), max(sleep_disturbances)))

        
        # 4. Sleep Duration
        print("4. Calculating sleep duration")
        sleep_schedules = get_all_circadian_rhythm(light_processed)

        del light_processed

        sleep_durations = get_sleep_durations(sleep_schedules)
        pp.pprint(sleep_durations)
        
        sleep_durations_normalised = []
        for val in sleep_durations:
            if val == -2:
                sleep_durations_normalised.append(-2)
            else:
                sleep_durations_normalised.append(normalise(val, min([x for x in sleep_durations if x != -2]), max(sleep_durations)))

                
        # Ranking
        print("Ranking results...")
        health_insurance_all = health_insurance(ages, microwave_normalised, stovetop_normalised, sleep_disturbances_normalised, sleep_durations_normalised)
        
        result = normalise_and_rank(health_insurance_all, list(light_circuits.keys()))
        
        for key in list(result.keys()):
            try:
                final_result[key] = final_result[key] + result[key]
            except KeyError:
                final_result[key] = result[key]
        print("Current result: ", final_result)
        print()

        gc.collect()
    
    for key in list(final_result.keys()):
        final_result[key] = final_result[key] // (len(months)-1)
    
    print("--- Final result ---\n", final_result)        
    return final_result

# The main function for processing the outliers calculation. 
# Iterating through the months of the sample, it reads from the db, calls the helper functions above, and ranks the households against each other.
def process_outlier_check():
    months = ['2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01']
    
    outliers = {}
    
    for j in range(len(months)-1):
        print("--- Processing {} => {} ---".format(months[j], months[j+1]))
        
        # Consumption Data
        print("Getting consumption data")
        consumption_dataframes = []
        for i, home in enumerate(power_circuits):
            consumption_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(power_circuits[home])))
        
        # Preprocessing
        print("Processing consumption data")
        consumption_processed = pre_process(consumption_dataframes)
        
        
        # 1. Outlier Checks
        print("1. Detecting outliers...")
        peaks = []
        for household in consumption_processed:
            curr_peaks = get_peaks(household, 4000, 20000, 120, 720, 0)
            peaks.append(curr_peaks)
        
        # Save outliers
        for i, key in enumerate(list(power_circuits.keys())):
            try:
                outliers[key] = peaks[i]
            except KeyError:
                outliers[key] = peaks[i]
        print("Current result: ")
        pp.pprint(list(outliers.values()))
        print()
    
    print("--- Outliers ---\n", outliers)        
    return outliers

# Gets the total W used in the data.
def get_solar(data):
    total = 0
    for val in data:
        total += val
    print(total)
    return int((total / len(data)) * (60*60*24/10))


# Gets all this data for all households.
def get_all_solars(data):
    consumptions = []
    for household in data:
        consumptions.append(get_solar(household))
    return consumptions

# Takes in the normalised values of all the variables of measuring sustainability and returns a rank to be displayed.
def sustainability(consumptions, solars):

    # For the households without solar, a low consumption means more sustainable.
    c_ranks = {}
    for i, home in enumerate(noSolar):
        if consumptions[i] != -2: 
            if (consumptions[i] < -0.8):
                curr = 3
            elif (consumptions[i] < -0.5):
                curr = 2
            elif (consumptions[i] < -0.2):
                curr = 1
            elif (consumptions[i] > 0.8):
                curr = -3
            elif (consumptions[i] > 0.5):
                curr = -2
            elif (consumptions[i] > 0.2):
                curr = -1
            try:
                c_ranks[home] = c_ranks[home] + curr
            except KeyError:
                c_ranks[home] = curr
    
    # For the households with solar, a high solar production means more sustainable.
    s_ranks = {}
    for i, home in enumerate(hasSolar):
        if solars[i] != -2: 
            if (solars[i] > 0.8):
                curr = -3
            elif (solars[i] > 0.5):
                curr = -2
            elif (solars[i] > 0.2):
                curr = -1
            elif (solars[i] < -0.8):
                curr = 3
            elif (solars[i] < -0.5):
                curr = 2
            elif (solars[i] < -0.2):
                curr = 1
            try:
                s_ranks[home] = s_ranks[home] + curr
            except KeyError:
                s_ranks[home] = curr
    
    print("Sustainability Ranks: \t\t\t", c_ranks, s_ranks)
    print()
    return list(c_ranks.values()), list(s_ranks.values())

# Represents households with/without solar so that there sustainability can be ranked respectively.
hasSolar = ["uq10", "uq12", "uq26", "uq56", "uq61", "uq67", "uq68", "hfs01a"]
noSolar = ["uq23", "uq24", "uq33", "uq37", "uq45", "uq48", "uq49", "uq57", "uq75", "uq85", "uq88", "uq92"]

# The main function for processing the sustainability calculation. 
# Iterating through the months of the sample, it reads from the db, calls the helper functions above, and ranks the households against each other.
def process_sustainability():
    months = ['2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01']

    final_result = {}

    print("\n------------- SUSTAINABILITY -------------\n")

    # print("1. Heat Sensitivity")
    # heat_sensitivity = process_heat_sensitivity()

    # print("2. Cold Sensitivity")
    # cold_sensitivity = process_cold_sensitivity()

    # for home in list(heat_sensitivity.keys()):
    #     final_result[home] = (heat_sensitivity[home] + cold_sensitivity[home]) // 2

    for j in range(len(months)-1):
        print("--- Processing {} => {} ---".format(months[j], months[j+1]))

        # Consumption Data
        print("Getting consumption data")
        consumption_dataframes = []
        for i, home in enumerate(noSolar):
            consumption_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'Consumption'"))
        
        # Preprocessing
        print("Processing consumption data")
        consumption_processed = pre_process(consumption_dataframes)

        del consumption_dataframes

        # 1. Consumption
        print("1. Calculating consumption")
        consumptions = get_all_consumptions(consumption_processed)
        print("Consumptions: ", consumptions)

        del consumption_processed
        
        consumption_normalised = []
        for val in consumptions:
            if val == -2:
                consumption_normalised.append(-2)
            else:
                consumption_normalised.append(normalise(val, min([x for x in consumptions if x != -2]), max(consumptions)))

        # Solar Data
        print("Getting solar data")
        solar_dataframes = []
        for i, home in enumerate(hasSolar):
            solar_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'Net'"))
        
        # Preprocessing
        print("Processing solar data")
        solar_processed = pre_process(solar_dataframes)

        del solar_dataframes

        # 2. Solar
        print("2. Calculating solar")
        solars = get_all_solars(solar_processed)
        print("Solars: ", solars)

        del solar_processed
        
        solar_normalised = []
        for val in solars:
            if val == -2:
                solar_normalised.append(-2)
            else:
                solar_normalised.append(normalise(val, min([x for x in solars if x != -2]), max(solars)))

        # Ranking
        print("Ranking results...")
        consumption_ranks, solar_ranks = sustainability(consumption_normalised, solar_normalised)
        
        c_result = normalise_and_rank(consumption_ranks, noSolar)
        s_result = normalise_and_rank(solar_ranks, hasSolar)
        print(c_result)
        print(s_result)

        result = {}

        for i, home in enumerate(noSolar):
            result[home] = c_result[home]
        
        for i, home in enumerate(hasSolar):
            result[home] = s_result[home]

        print("This result: ", result)
        pp.pprint(list(result.values()))
        print()
        
        for i, key in enumerate(list(aircon_circuits.keys())):
            try:
                final_result[key] = final_result[key] + result[key]
            except KeyError:
                final_result[key] = result[key]

        gc.collect()
    
    for key in list(final_result.keys()):
        final_result[key] = final_result[key] // (len(months)-1)
        
    print("--- Final result ---\n", final_result)        
    return final_result



# Deprecated
# The main function for processing the heat sensitivity calculation. 
# Iterating through the months of the sample, it reads from the db, calls the helper functions above, and ranks the households against each other.
def process_heat_sensitivity():

    months = ['2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01']
    bom_data_max = load_bom_data("data/extra/bom_data_max.csv")

    final_result = {}

    print("\n------------- HEAT SENSITIVITY -------------\n")

    for j in range(len(months)-1):
        print("--- Processing {} => {} ---".format(months[j], months[j+1]))

        print("Getting aircon data")
        aircon_dataframes = []
        for i, home in enumerate(aircon_circuits):
            if (aircon_circuits[home] == "none"):
                aircon_dataframes.append(pd.DataFrame())
            else:
                aircon_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(aircon_circuits[home])))
        
        # Preprocessing
        print("Processing aircon data")
        aircon_processed = pre_process(aircon_dataframes)

        del aircon_dataframes

        aircon_peaks = []
        for df in aircon_processed:
            aircon_peaks.append(get_peaks(df, 100, 10000, 10, 1000, 60))

        watt_hours = []
        for i, df in enumerate(aircon_processed):
            watt_hours.append(get_watt_hours(aircon_peaks[i], df))

        del aircon_processed

        usages = []
        for watt_hour in watt_hours:
            if (len(watt_hour) != 0):
                usages.append(get_daily_usage(watt_hour).tolist())
            else:
                usages.append(-100)

        # TODO: Maybe instead of correlation, simply use the probability of aircon use given a certain temperature?
        # calculate the Pearson's correlation between two variables
        seed(1)

        correlations = []
        print("BOM: ", bom_data_max[j])
        print("USAGES: \n")
        for usage in usages:
            print(usage)
            if (usage != -100):
                # Pearsonr measures the correlation between the input variables.
                # Positive correlation: As variable A goes up, variable B goes up. 1 is perfect positive correlation.
                # Negative correlation: As variable B goes up, variable B goes down. -1 is perfect negative correlation.
                # No correlation: There is no observable trend between variables A and B. 0 means the two variables are completely random.
                correlations.append(pearsonr(bom_data_max[j], usage)[0])
            else:
                correlations.append(-2)
        pp.pprint(correlations)

        correlations_normalised = []
        for val in correlations:
            if val == -2:
                correlations_normalised.append(-9)
            else:
                correlations_normalised.append(normalise(val, min([x for x in correlations if x != -9]), max(correlations)))


        # Ranking
        print("Ranking results...")
        heat_sensitivity_all = heat_sensitivity(correlations_normalised)
        
        result = normalise_and_rank(heat_sensitivity_all, list(aircon_circuits.keys()))
        
        for i, key in enumerate(list(aircon_circuits.keys())):
            try:
                final_result[key] = final_result[key] + result[key]
            except KeyError:
                final_result[key] = result[key]
        print("Current result: ", final_result)
        print()

        gc.collect()
    
    for key in list(final_result.keys()):
        final_result[key] = final_result[key] / (len(months)-1)
        
    print("--- Final result ---\n", final_result)        
    return final_result

# Deprecated
# The main function for processing the cold sensitivity calculation. 
# Iterating through the months of the sample, it reads from the db, calls the helper functions above, and ranks the households against each other.
def process_cold_sensitivity():

    months = ['2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01']
    bom_data_min = load_bom_data("data/extra/bom_data_min.csv")

    final_result = {}
    
    print("\n------------- COLD SENSITIVITY -------------\n")
    
    for j in range(len(months)-1):
        print("--- Processing {} => {} ---".format(months[j], months[j+1]))

        print("Getting aircon data")
        aircon_dataframes = []
        for i, home in enumerate(aircon_circuits):
            if (aircon_circuits[home] == "none"):
                aircon_dataframes.append(pd.DataFrame())
            else:
                aircon_dataframes.append(get_data_in_period(client, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(aircon_circuits[home])))
        
        # Preprocessing
        print("Processing aircon data")
        aircon_processed = pre_process(aircon_dataframes)

        del aircon_dataframes

        aircon_peaks = []
        for df in aircon_processed:
            aircon_peaks.append(get_peaks(df, 100, 10000, 10, 1000, 60))

        watt_hours = []
        for i, df in enumerate(aircon_processed):
            watt_hours.append(get_watt_hours(aircon_peaks[i], df))

        del aircon_processed

        usages = []
        for watt_hour in watt_hours:
            if (len(watt_hour) != 0):
                usages.append(get_daily_usage(watt_hour).tolist())
            else:
                usages.append(-100)

        # calculate the Pearson's correlation between two variables
        seed(1)

        correlations = []
        print("BOM: ", bom_data_min[j])
        print("USAGES: \n")
        for usage in usages:
            print(usage)
            if (usage != -100):
                # Pearsonr measures the correlation between the input variables.
                # Positive correlation: As variable A goes up, variable B goes up. 1 is perfect positive correlation.
                # Negative correlation: As variable B goes up, variable B goes down. -1 is perfect negative correlation.
                # No correlation: There is no observable trend between variables A and B. 0 means the two variables are completely random.
                correlations.append(pearsonr(bom_data_min[j], usage)[0])
            else:
                correlations.append(-2)
        pp.pprint(correlations)

        correlations_normalised = []
        for val in correlations:
            if val == -2:
                correlations_normalised.append(-9)
            else:
                correlations_normalised.append(normalise(val, min([x for x in correlations if x != -9]), max(correlations)))


        # Ranking
        print("Ranking results...")
        cold_sensitivity_all = cold_sensitivity(correlations_normalised)
        
        result = normalise_and_rank(cold_sensitivity_all, list(aircon_circuits.keys()))
        
        for i, key in enumerate(list(aircon_circuits.keys())):
            try:
                final_result[key] = final_result[key] + result[key]
            except KeyError:
                final_result[key] = result[key]
        print("Current result: ", final_result)
        print()

        gc.collect()
    
    for key in list(final_result.keys()):
        final_result[key] = final_result[key] / (len(months)-1)
        
    print("--- Final result ---\n", final_result)        
    return final_result

# Deprecated
# Takes in the normalised values of all the variables of measuring heat sensitivity and returns a rank to be displayed.
def heat_sensitivity(correlations):
    print("Correlations: ", correlations)

    ranks = [0] * len(correlations)
    for i, correlation in enumerate(correlations):
        if correlation != -9:
            if (correlation >= 0.8):
                ranks[i] += 4
            elif (correlation >= 0.6):
                ranks[i] += 3
            elif (correlation >= 0.4):
                ranks[i] += 2
            elif (correlation >= 0.2):
                ranks[i] += 1
            elif (correlation <= -0.8):
                ranks[i] -= 4
            elif (correlation <= -0.6):
                ranks[i] -= 3
            elif (correlation <= -0.4):
                ranks[i] -= 2
            elif (correlation <= -0.2):
                ranks[i] -= 1
    
    print("Heat Sensitivity Ranks: \t\t\t", ranks)
    print()
    return ranks

# Deprecated
# Takes in the normalised values of all the variables of measuring cold sensitivity and returns a rank to be displayed.
def cold_sensitivity(correlations):
    print("Correlations: ", correlations)

    ranks = [0] * len(correlations)
    for i, correlation in enumerate(correlations):
        if correlation != -9:
            if (correlation >= 0.8):
                ranks[i] -= 4
            elif (correlation >= 0.6):
                ranks[i] -= 3
            elif (correlation >= 0.4):
                ranks[i] -= 2
            elif (correlation >= 0.2):
                ranks[i] -= 1
            elif (correlation <= -0.8):
                ranks[i] += 4
            elif (correlation <= -0.6):
                ranks[i] += 3
            elif (correlation <= -0.4):
                ranks[i] += 2
            elif (correlation <= -0.2):
                ranks[i] += 1
    
    print("Cold Sensitivity Ranks: \t\t\t", ranks)
    print()
    return ranks


def gen_subplots_for_event():
    granularities = [1/6, 1, 5, 15, 60]
    granularities_str = ["10sec", "1min", "5min", "15min", "1h"]
    title = "Aircon"
    household = 'hfs01a'
    circuit = 'Aircon1'
    start = '2021-09-04T00:00:00Z'
    end = '2021-09-11T00:00:00Z'
    min_power = 800
    max_power = 3500
    min_duration = 1
    max_duration = 360
    off_requirement = 30

    client = InfluxDBClient(host='live2.phisaver.com', database='phisaver', username='reader', password='Rmagine!', port=8086, headers={'Accept': 'application/json'}, gzip=True)
    dataframes = get_data_in_period(client, "'{}'".format(start), "'{}'".format(end), "'{}'".format(household), "'{}'".format(circuit))
    fig, axs = plt.subplots(len(granularities), figsize=(14, 8), sharey=True, tight_layout=True)

    for i in range(len(granularities)):
        print("Processing granularity: {}".format(granularities_str[i]))
        processed = pre_process(dataframes, granularities[i])
        for df in processed:
            peaks = get_peaks(df, min_power, max_power, min_duration, max_duration, off_requirement, granularities[i])
            peaks_dates = [i[0] for i in peaks]
            peaks_values = [i[1] for i in peaks]

            x = [i for i in df.index.tolist()]
            y = [i for i in df.values.tolist()]

            axs[i].axhline(y=max_power, color='c', linestyle='--', label="Upper bound")
            axs[i].axhline(y=min_power, color='g', linestyle='--', label="Lower bound")

            locs = list(range(0, len(x), round(len(x)/7)))
            labels = [datetime.datetime.strptime(x[i], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d") for i in locs]
            plt.sca(axs[i])
            plt.xticks(locs, labels)

            axs[i].plot(x, y, label="Circuit Data", zorder=1)
            axs[i].scatter(peaks_dates, peaks_values, c="r", marker='x', label="Peaks Detected", zorder=2)
            axs[i].text(0.5, 1.03, "{}".format(granularities_str[i]), transform=axs[i].transAxes, ha="center")

            # axs[i].legend(loc=1)
            axs[i].text(0.1, 0.75, "{} peaks".format(len(peaks)), transform=axs[i].transAxes)

            axs[i].margins(x=0)

    axs[0].legend(loc=1)
    for i in range(len(axs)-1):
        axs[i].set_xticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(bottom=0.5)
    plt.tight_layout()

    fig.suptitle("Detection of {} Usage with Data of Varying Granularity (mean aggregation)".format(title))
    fig.text(0.5, 0, 'Time', ha='center')
    fig.text(0, 0.5, 'Power (Watts)', va='center', rotation='vertical')
    print("Generating plots...")   
    plt.show()


def gen_subplots_for_microwave():
    granularities = [1/6, 1, 5, 15, 60]
    granularities_str = ["10sec", "1min", "5min", "15min", "1h"]
    title = "Microwave"
    household = 'hfs01a'
    circuit = 'Power1'
    start = '2021-09-08T00:00:00Z'
    end = '2021-09-11T00:00:00Z'
    min_power = 600
    max_power = 1200
    min_duration = 1
    max_duration = 10
    off_requirement = 5

    client = InfluxDBClient(host='live2.phisaver.com', database='phisaver', username='reader', password='Rmagine!', port=8086, headers={'Accept': 'application/json'}, gzip=True)
    dataframes = get_data_in_period(client, "'{}'".format(start), "'{}'".format(end), "'{}'".format(household), "'{}'".format(circuit))
    fig, axs = plt.subplots(len(granularities), figsize=(14, 8), sharey=True, tight_layout=True)

    for i in range(len(granularities)):
        print("Processing granularity: {}".format(granularities_str[i]))
        processed = pre_process(dataframes, granularities[i])
        for df in processed:
            peaks = get_peaks(df, min_power, max_power, min_duration, max_duration, off_requirement, granularities[i])
            peaks_dates = [i[0] for i in peaks]
            peaks_values = [i[1] for i in peaks]

            x = [i for i in df.index.tolist()]
            y = [i for i in df.values.tolist()]

            axs[i].axhline(y=max_power, color='c', linestyle='--', label="Upper bound")
            axs[i].axhline(y=min_power, color='g', linestyle='--', label="Lower bound")

            locs = list(range(0, len(x), round(len(x)/5))) # Maybe just set the xticks to be whenener it is midnight on a given day.
            labels = [datetime.datetime.strptime(x[i], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %Hh") for i in locs]
            plt.sca(axs[i])
            plt.xticks(locs, labels)

            axs[i].plot(x, y, label="Circuit Data", zorder=1)
            axs[i].scatter(peaks_dates, peaks_values, c="r", marker='x', label="Peaks Detected", zorder=2)
            axs[i].text(0.5, 1.03, "{}".format(granularities_str[i]), transform=axs[i].transAxes, ha="center")

            # axs[i].legend(loc=1)
            axs[i].text(0.1, 0.75, "{} peaks".format(len(peaks)), transform=axs[i].transAxes)

            axs[i].margins(x=0)

    axs[0].legend(loc=1)
    for i in range(len(axs)-1):
        axs[i].set_xticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(bottom=0.5)
    plt.tight_layout()

    fig.suptitle("Detection of {} Usage with Data of Varying Granularity (mean aggregation)".format(title))
    fig.text(0.5, 0, 'Time', ha='center')
    fig.text(0, 0.5, 'Power (Watts)', va='center', rotation='vertical')
    print("Generating plots...")   
    plt.show()

# Main function
if __name__ == '__main__':
    # start = datetime.datetime.now()

    # gen_subplots_for_microwave()
    household = 'hfs01a'
    circuit = 'Power1'
    start = '2021-09-08T00:00:00Z'
    end = '2021-09-11T00:00:00Z'

    client = InfluxDBClient(host='54.253.86.187', database='phisaver', username='reader', password='Rmagine!', port=8086, headers={'Accept': 'application/json'}, gzip=True)

    dataframes = []
    dataframes.append(get_data_in_period(client, "'{}'".format(start), "'{}'".format(end), "'{}'".format(household), "'{}'".format(circuit)))

    processed = pre_process(dataframes)


    pp.pprint(processed)

    # end = datetime.datetime.now()

    # print()
    # print("Start: ", start)
    # print("End: ", end)

    # print("Time taken: {} minutes".format(end.time().minute - start.time().minute))

# %%
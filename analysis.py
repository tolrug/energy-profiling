#!/usr/bin/env python
# coding: utf-8

import sys
import datetime
import math
from typing import List, Tuple
import numpy as np
from numpy.lib.function_base import corrcoef
import pandas as pd
import matplotlib.pyplot as plt
import pprint as pp
import gc
pd.options.mode.chained_assignment = None  # default='warn'
# import ipdb; ipdb.set_trace()
pd.set_option('display.max_rows', 100)
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr

# df = pd.read_csv("data/steve-mar-2021.csv", sep=';', index_col=0, dtype=object)

# df = df.sort_values(by='Time', ascending=True)

# df = df.iloc[1:,:]

# powerpoints_1=df.loc[:, "Power1"]

def filter_function(x):
    if type(x) == str:
        return 0
    else:
        return int(round(float(x),0))
    
# Removes invalid data and converts everything to whole number-rounded ints
def process_row(series: pd.Series) -> pd.Series:
    series1 = pd.Series([0 if str(row).startswith("'") else row for row in series])
    series1 = series1.apply(lambda row: int(round(float(row),0)))
    series1.index = series.index
    return series1

# powerpoints_1 = process_row(powerpoints_1)


# In[4]:


# Returns a tuple of shape (closest_index, closest_value),
# which contains the closest value to the given value in the given array
def get_nearest(arr: List[int], value: int) -> Tuple[int, int]:
    closest = sys.maxsize
    closest_index = sys.maxsize
    for index, i in enumerate(arr):
        if (abs(value-i) < abs(value-closest)):
            closest = i
            closest_index = index
    
    return closest_index, closest


# In[5]:


# Returns an ordered series where the indexes are power values and the first values are the frequencies.
# The first element in the series is the most frequent 
def get_frequencies(freqs: pd.Series, series: pd.Series) -> pd.Series:
#     freqs = cleaned_data.value_counts()
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

        
        
        
        
        


# In[ ]:


# freqs = powerpoints_1.value_counts()


# In[ ]:


# first_freqs_count = get_frequencies(freqs, powerpoints_1)


# In[ ]:


# Calling this a second time (with the result of the first call passed in) groups the frequencies more tightly
# second_freqs_count = get_frequencies(first_freqs_count, powerpoints_1)


# In[ ]:


# second_freqs_count


# In[6]:


# Gets the two most frequent power readings of the data.
# From this, we can determine the cycling power signals of the refrigerator.
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
    for index, freq in value_counts.iteritems():
        if (not (abs(ser[i] - mode) < 10)):
            second_mode = ser[i]
            break
        i+=1

    return mode, second_mode
        
    
    


# In[ ]:


# get_modes(powerpoints_1)


# In[7]:


# Return True if the peak value (less the peak start value) is within the range specified (less the peak start value)
def isInRange(peak_value: int, trough_value: int, min_power: int, max_power: int) -> bool:
    if ((peak_value - trough_value >= min_power) and
        (peak_value - trough_value <= max_power)):
        return True
    return False


# In[8]:


# Returns a series of tuples of shape: (peak_start_time, peak_max_value, peak_duration),
# where each peak is within the range specified, finishes within the specified duration, and 
# is off for at least off_requirement minutes after the peak
def get_peaks(data: pd.Series, min_power: int, max_power: int, min_required_duration: int = 0, max_required_duration: int = sys.maxsize, off_requirement: int = 0) -> List[Tuple[str, int, int]]:
    i = 5
    peaks = []
    
    while i < len(data):
#         if (i == 3916):
#             import ipdb; ipdb.set_trace()

        peak_duration = 2 # +1 for the initial turning on, +1 for turning off
        try:
            curr_idx = i
            prev_idx = i-1
            next_idx = i+1
            climb_start_idx = 0
            peak_idx = 0
            if (data.iloc[curr_idx] - data.iloc[prev_idx] > 50): # could be the start of a peak
                climb_start_idx = prev_idx # mark start of climb...
                peak_idx = curr_idx
                while(True):
                    curr_off_duration = off_requirement
                    if (data.iloc[next_idx] - data.iloc[curr_idx] > 10): # still rising... not at peak yet
                        curr_idx+=1
                        prev_idx+=1
                        next_idx+=1
                        if (data.iloc[curr_idx] > data.iloc[peak_idx]):
                            peak_idx=curr_idx
                        peak_duration+=1
                    elif (abs(data.iloc[next_idx] - data.iloc[curr_idx]) < 10): # not rising but still at the same peak
                        curr_idx+=1
                        prev_idx+=1
                        next_idx+=1
                        peak_duration+=1
                    elif (data.iloc[next_idx] > data.iloc[climb_start_idx] + 150): # not at (close enough to) being a trough yet
                        curr_idx+=1
                        prev_idx+=1
                        next_idx+=1
                        peak_duration+=1
                    else: # at a trough, so append climb_start which is the initial peak
                        if (peak_duration <= max_required_duration and 
                            isInRange(data.iloc[peak_idx], data.iloc[climb_start_idx], min_power, max_power)):
                            satisfied_off_requirement = True
                            while (curr_off_duration > 0):
                                if (data.iloc[next_idx+1] > data.iloc[next_idx] + 100): # hasn't been a trough for long enough
                                    satisfied_off_requirement = False
                                    break
                                curr_off_duration -= 1
                                curr_idx+=1
                                prev_idx+=1
                                next_idx+=1
                            if (not satisfied_off_requirement):
                                curr_idx = next_idx
                                prev_idx = curr_idx-1
                                next_idx+=1
                                continue
                            if (not peak_duration < min_required_duration):
                                peaks.append((data.index[climb_start_idx], data.iloc[peak_idx] - data.iloc[climb_start_idx], peak_duration))
                        break
            i = next_idx
        except IndexError:
            i = next_idx
    
    return peaks


# In[ ]:


# food_prep_peaks = get_peaks(powerpoints_1, 700, 2200, 0, 10, 0)
# print(len(food_prep_peaks))
# pp.pprint(food_prep_peaks)


# # In[ ]:


# hfs01a_microwave_peaks = get_peaks(powerpoints_1, 600, 1200, 0, 10, 0)
# print(len(hfs01a_microwave_peaks))
# pp.pprint(hfs01a_microwave_peaks)


# In[ ]:


# Stovetop peaks
# stove=df.loc[:, "Oven"]
# stove = process_row(stove)
# stove


# # In[ ]:


# hfs01a_stove_peaks = get_peaks(stove, 500, 5000, 0, 120, 60)
# print(len(hfs01a_stove_peaks))
# pp.pprint(hfs01a_stove_peaks)


# # In[ ]:


# lights=df.loc[:, "Lights2"]


# # In[ ]:


# lights = process_row(lights)


# In[ ]:


# get_modes(lights)


# In[9]:


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
    
    
    


# In[10]:


# Returns the average time in the given data as a formatted string
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
        
        
        
    


# In[11]:


# Returns the most common sleep time and wake time (i.e. circadian rythm)
# TODO: Ignore weekends for a more accurate result
def get_circadian_rythm(data: pd.Series) -> Tuple[str, str]:
    i = 0
    res = []
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
            
            if on_duration > 120 and timeInRange(data.index[i][11:], '19:00:00', '02:00:00', True): # lights have been on for more than 100 minutes
                off_cases.append((data.index[i], 'turning off', on_duration))

            while (data.iloc[i] <= resting_power): # lights are off
                off_duration += 1
                i+=1
                
            if off_duration > 120 and timeInRange(data.index[i][11:], '03:00:00', '10:00:00', False): # lights have been off for more than 100 minutes
                on_cases.append((data.index[i], 'turning on', off_duration))

            i+=1
        
        except IndexError:
            i+=1
            
    average_wake = get_average_wake_or_sleep(on_cases)
    average_sleep = get_average_wake_or_sleep(off_cases, True)
    
    return (average_wake, average_sleep)


# In[12]:


# Returns the most common left the house and got home from work (i.e. work schedule)
# TODO: Ignore weekends for a more accurate result
def get_work_schedule(data: pd.Series) -> Tuple[str, str]:
    i = 0
    res = []
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
            
            if on_duration > 120 and timeInRange(data.index[i][11:], '05:00:00', '10:00:00', False): # lights have been on for more than 100 minutes
                off_cases.append((data.index[i], 'turning off', on_duration))

            while (data.iloc[i] <= resting_power): # lights are off
                off_duration += 1
                i+=1
                
            if off_duration > 120 and timeInRange(data.index[i][11:], '14:00:00', '19:00:00', False): # lights have been off for more than 100 minutes
                on_cases.append((data.index[i], 'turning on', off_duration))

            i+=1
        
        except IndexError:
            i+=1
            
    average_home = get_average_wake_or_sleep(on_cases)
    average_leave = get_average_wake_or_sleep(off_cases, True)
    
    return (average_leave, average_home)


# In[ ]:


# circadian_rythm = get_circadian_rythm(lights)
# circadian_rythm


# # In[ ]:


# aircon = df.loc[:, "Aircon1"]
# aircon = process_row(aircon)

    


# # In[ ]:


# aircon_peaks = get_peaks(aircon, 100, 10000, 10, 1000, 60)
# print(len(aircon_peaks))
# pp.pprint(aircon_peaks)


# In[13]:


# Returns the energy usage (Wh) of each peak
def get_watt_hours(peaks: List[Tuple[str,int,int]], data: pd.Series) -> List[Tuple[str,int]]:
    watt_hours = []
    for peak in peaks:
        watt_hour = 0
        for _ in range(peak[2]):
            watt_hour += data.loc[peak[0]]
        watt_hours.append((peak[0], watt_hour))

    return watt_hours


# In[ ]:


# watt_hours = get_watt_hours(aircon_peaks, aircon)
# watt_hours


# In[14]:


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
    from calendar import monthrange
    days_in_month = monthrange(int(df.index[-1].strftime("%Y")), int(df.index[-1].strftime("%m")))[1]
    while (i <= days_in_month):
        datetime = pd.Timestamp(int(df.index[-1].strftime("%Y")), int(df.index[-1].strftime("%m")), i, 0, 0, 0)
        df.loc[datetime] = 0
        df.sort_index(inplace=True) 
        i+=1
        
    return df


# In[ ]:


# usage_series = get_daily_usage(watt_hours)
# usage_series


# In[ ]:


def load_bom_data(filename: str) -> List[int]:
    all_data = pd.read_csv(filename, sep=',', header=None, dtype=object)
    data = []
    for i in range(6):
        month = all_data.iloc[:, i]
        month = month.dropna()
        data.append([int(float(month.iloc[j])) if month.iloc[j] != "NaN" else True for j in range(len(month))])
    return data


# In[ ]:


# mar_2021 = load_bom_data("data/extra/bom-mar-2021.csv", True)

# dec_2020 = load_bom_data("data/extra/bom-dec-2020.csv", True)
# jan_2021 = load_bom_data("data/extra/bom-jan-2021.csv", True)
# feb_2021 = load_bom_data("data/extra/bom-feb-2021.csv", True)
# max_temps_summer = dec_2020 + jan_2021 + feb_2021

# jun_2020 = load_bom_data("data/extra/bom-jun-2020.csv", False)
# jul_2020 = load_bom_data("data/extra/bom-jul-2020.csv", False)
# aug_2020 = load_bom_data("data/extra/bom-aug-2020.csv", False)
# min_temps_winter = jun_2020 + jul_2020 + aug_2020


# In[ ]:


# df = pd.read_csv("data/uq49_01-06-2020--31-08-2020.csv", sep=';', index_col=0, dtype=object)
# df = df.sort_values(by='Time', ascending=True)
# df = df.iloc[1:,:]

# df1 = pd.read_csv("data/uq49_01-12-2020--28-02-2021.csv", sep=';', index_col=0, dtype=object)
# df1 = df1.sort_values(by='Time', ascending=True)
# df1 = df1.iloc[1:,:]

# df2 = pd.read_csv("data/uq49_01-12-2020--31-12-2020.csv", sep=';', index_col=0, dtype=object)
# df2 = df2.sort_values(by='Time', ascending=True)
# df2 = df2.iloc[1:,:]


# # In[ ]:


# powerpoints_uq49 = df2.loc[:, "Powerpoints1"]
# powerpoints_uq49 = process_row(powerpoints_uq49)

# hob_uq49 = df2.loc[:, "Hob"]
# hob_uq49 = process_row(hob_uq49)

# microwave_peaks_uq49 = get_peaks(powerpoints_uq49, 600, 1200, 0, 10, 0)
# print(len(microwave_peaks_uq49))
# pp.pprint(microwave_peaks_uq49)

# stove_peaks_uq49 = get_peaks(hob_uq49, 500, 5000, 0, 120, 60)
# print(len(stove_peaks_uq49))
# pp.pprint(stove_peaks_uq49)


# # In[ ]:


# aircon_winter = df.loc[:, "Aircon1"]
# aircon_winter = process_row(aircon_winter)

# aircon_summer = df1.loc[:, "Aircon1"]
# aircon_summer = process_row(aircon_summer)


# In[ ]:


# aircon_winter


# # In[ ]:


# aircon_summer


# # In[ ]:


# aircon_peaks_winter = get_peaks(aircon_winter, 100, 10000, 0, 1000, 60)
# aircon_peaks_winter


# In[ ]:


# aircon_peaks_summer = get_peaks(aircon_summer, 100, 10000, 0, 1000, 60)
# aircon_peaks_summer


# # In[ ]:


# watt_hours_winter = get_watt_hours(aircon_peaks_winter, aircon_winter)
# watt_hours_winter


# # In[ ]:


# watt_hours_summer = get_watt_hours(aircon_peaks_summer, aircon_summer)


# In[ ]:


# usage_series_winter = get_daily_usage(watt_hours_winter)
# usage_series_winter


# # In[ ]:


# usage_series_summer = get_daily_usage(watt_hours_summer)
# usage_series_summer


# In[ ]:


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


# In[ ]:


# plot_aircon_trend(usage_series, mar_2021, s_type="Max")


# # In[ ]:


# plot_aircon_trend(usage_series_winter, min_temps_winter, s_type="Min")


# # In[ ]:


# plot_aircon_trend(usage_series_summer, max_temps_summer, s_type="Max")


# In[ ]:


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
        


# In[ ]:


# get_probabilities(mar_2021, usage_series)


# In[ ]:





# In[ ]:


def plot_probability_curve(temps: List[int], usage_series: pd.Series):
        
    min_temp = min(temps)
    max_temp = max(temps)
        
    fig = plt.figure(figsize=(8,6))
        
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


# In[ ]:


# plot_probability_curve(mar_2021, usage_series)


# # In[ ]:


# plot_probability_curve(max_temps_summer, usage_series_summer)


# # In[ ]:


# plot_probability_curve(min_temps_winter, usage_series_winter)


# In[ ]:


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
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(all_temps, interpolated)

    ax = plt.gca()
    ax.set_ylim([0.0,1.0])

    plt.xlabel("Temperature (C)")
    plt.ylabel("Probability of Aircon Use")
    plt.title("Probability Curve of Aircon Usage vs Temperature")
    plt.show()
    
    return all_temps, all_probs


# In[ ]:


# TODO: Maybe filter this so that only major aircon usages are plotted.
# plot_probs_combined(min_temps_winter, usage_series_winter, max_temps_summer, usage_series_summer)


# In[ ]:





# In[ ]:


# Microwave vs Stovetop Visualisation

#  1. Get access to PhiSaver API
#  2. Get average of all households' microwave use AND stovetop use
#  3. ...


# In[ ]:





# In[14]:


# DATABASE INTERACTIONS START HERE ---------------------------------------------------------------------------------


# In[15]:


from influxdb import InfluxDBClient, DataFrameClient


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


client2 = InfluxDBClient(host='live2.phisaver.com', database='phisaver', username='reader', password='Rmagine!', port=8086, headers={'Accept': 'application/json'}, gzip=True)
# q = """SELECT * FROM "iotawatt" WHERE "device" = 'hfs01a' AND "time" > '2021-05-31T11:58:20Z' - 12w"""

# client_df = pd.DataFrame(client2.query(q, chunked=True))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[180]:


oven_circuits = {                     "uq10": "Oven",
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


# In[181]:


# Change this to be the circuit with the most kitchen-like activity (not neccessarily the fridge's circuit)
power_circuits = {                     
                    "uq10": "Powerpoints1",
                    "uq12": "Powerpoints2",
                    "uq23": "Powerpoints1",
                    "uq24": "Powerpoints2",
                    "uq26": "Powerpoints1",  
                    "uq33": "Powerpoints2",
                    "uq37": "PowerpointsRear",
                    "uq45": "Powerpoints1",
                    "uq48": "Powerpoints2",
                    "uq49": "Powerpoints1",
                    "uq56": "Powerpoints2",
                    "uq57": "Powerpoints4",
                    "uq61": "Misc1",
                    "uq67": "Powerpoints1", 
                    "uq68": "Powerpoints1",
                    "uq75": "Powerpoints1",
                    "uq85": "Powerpoints2",
                    "uq88": "Powerpoints1",
                    "uq92": "Powerpoints1",
                    "hfs01a": "Power1",
                 }


# In[18]:


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


# In[ ]:


# Calculate average (or similar) and bounds for microwave and stovetop usage


# In[187]:


def get_data_in_period(client, start_date, end_date, device, sensor):
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


# In[188]:


def get_most_recent_time(df):
    last_idx = df.index[-1]
    last_arr = df[0][last_idx]
    if (last_arr is None):
        return -1
    last_time = last_arr[-1].get("time")
    return "'{}'".format(last_time)


# In[37]:


# microwave_dataframes = []
# for i, home in enumerate(power_circuits):
#     microwave_dataframes.append(get_data_in_period(client2, "'2021-03-01T00:00:00Z'", "'2021-03-31T23:59:50Z'", "'{}'".format(home), "'{}'".format(power_circuits[home])))


# # In[ ]:


# stove_dataframes = []
# for i, home in enumerate(stove_circuits):
#     stove_dataframes.append(get_data_in_period(client2, "'2020-10-01T00:00:00Z'", "'2021-03-31T23:59:50Z'", "'{}'".format(home), "'{}'".format(stove_circuits[home])))    


# In[ ]:





# In[ ]:


# Now pre-process these df's into the format required for the process() method.


# In[106]:


def pre_process(dataframes):
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
            to_append = to_append.set_index('time')
            to_append = to_append.loc[:, "Watts"]
#             to_append = to_append[~to_append.index.duplicated()]
            to_append = to_append.to_frame()
    
            # Filter out the time values that are not on the minute for getting 1-min granularity
#             to_append = to_append.groupby(np.arange(len(to_append))//6).mean()
#             to_append = to_append.loc[to_append.index.str.endswith('00Z')]
            df = df.append(to_append)
        
        processing = process_row(df.squeeze())
        # Granularity filter
#         print("PPPP: ", processing)
#         processing_means = processing.groupby(np.arange(len(processing))//6).sum()
#         processing_means.index = processing.index[::6]

        post_process.append(processing)
        
    for df in post_process:
        df.index = df.index.to_series().apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S'))

    return post_process


# In[ ]:


# type(microwave_dataframes)


# # In[ ]:


# processed = pre_process(microwave_dataframes)
# processed


# # In[ ]:


# stove_processed = pre_process(stove_dataframes)
# stove_processed


# In[ ]:





# In[22]:


def microwave_stove_peaks(processed, stove_processed):
    microwave = []
    stove = []
    
    for df in processed:
        microwave_peaks = get_peaks(df, 600, 1200, 0, 10, 0)
        microwave.append(len(microwave_peaks))
        
    for df in stove_processed:
        stove_peaks = get_peaks(df, 500, 5000, 0, 120, 60)
        stove.append(len(stove_peaks))
        
    return microwave, stove


# In[ ]:


# microwave_peaks, stove_peaks = microwave_stove_peaks(processed, stove_processed)


# # In[ ]:


# print(microwave_peaks)
# print(stove_peaks)


# In[23]:


def normalise(x, x_min, x_max):
    try:
        return round(2*((x-x_min)/(x_max-x_min))-1,2)
    except ZeroDivisionError:
        return 0


# In[ ]:


# uq49_microwave_normalised = normalise(len(microwave_peaks_uq49), min(microwave_peaks), max(microwave_peaks))
# uq49_stove_normalised = normalise(len(stove_peaks_uq49), min(stove_peaks), max(stove_peaks))

# print(uq49_microwave_normalised)
# print(uq49_stove_normalised)


# In[ ]:


# hfs01a_microwave_normalised = normalise(len(hfs01a_microwave_peaks), min(microwave_peaks), max(microwave_peaks))
# hfs01a_stove_normalised = normalise(len(hfs01a_stove_peaks), min(stove_peaks), max(stove_peaks))

# print(hfs01a_microwave_normalised)
# print(hfs01a_stove_normalised)


# In[ ]:


# import requests
# # data={'microwave': hfs01a_microwave_normalised, 'stovetop': hfs01a_stove_normalised}
# data={'microwave': uq49_microwave_normalised, 'stovetop': uq49_stove_normalised}

# r = requests.post('http://localhost:3001/data', data=data)
# print(r.json())
# print(r.status_code)


# In[ ]:


# %history -g


# In[182]:

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

# light_circuits = {
#                 "uq10": "Lights1",
#                 "uq12": "Lights12",
#                 "uq23": "Lights1",
#                 "uq24": "Lights1",
#                 "uq26": "Light",
#                 "hfs01a": "Lights2",
#                 }


# In[ ]:


# !!!  Only call this when needing to query DB again  !!!

# light_dataframes = []
# for i, home in enumerate(light_circuits):
#     light_dataframes.append(get_data_in_period(client2, "'2020-10-01T00:00:00Z'", "'2021-03-31T23:59:50Z'", "'{}'".format(home), "'{}'".format(light_circuits[home])))


# # In[ ]:


# type(light_dataframes)


# # In[ ]:


# light_processed = pre_process(light_dataframes)
# light_processed


# In[25]:


def get_all_circadian_rythm(light_processed):
    lst = []
    for i in range(len(light_processed)):
        try:
            lst.append(get_circadian_rythm(light_processed[i]))
        except ZeroDivisionError:
            print("zero div err")
            lst.append(0)
    return lst


# In[ ]:


# circadian_rythms = get_all_circadian_rythm(light_processed)
# circadian_rythms


# In[ ]:





# In[ ]:


# def get_average_circadian_rythm_time(times):
#     total_morning_seconds = 0
#     total_night_seconds = 0
#     min_morning = '23:59:59'
#     max_morning = '00:00:01'
#     min_night = '23:59:59'
#     max_night = '00:00:01'
    
#     fmt = "%H:%M:%S"
    
#     for tup in times:
#         total_morning_seconds += int(tup[0][6:8])
#         total_morning_seconds += int(tup[0][3:5]) * 60
#         total_morning_seconds += int(tup[0][0:2]) * 60 * 60
        
#         total_night_seconds += int(tup[1][6:8])
#         total_night_seconds += int(tup[1][3:5]) * 60
#         total_night_seconds += int(tup[1][0:2]) * 60 * 60
        
#         if (datetime.datetime.strptime(tup[0], fmt) < datetime.datetime.strptime(min_morning, fmt)):
#             min_morning = tup[0]
#         if (datetime.datetime.strptime(tup[0], fmt) > datetime.datetime.strptime(max_morning, fmt)):
#             max_morning = tup[0]
            
#         if (datetime.datetime.strptime(tup[1], fmt) < datetime.datetime.strptime(min_night, fmt)):
#             min_night = tup[1]
#         if (datetime.datetime.strptime(tup[1], fmt) > datetime.datetime.strptime(max_night, fmt)):
#             max_night = tup[1]
        
#     seconds = total_morning_seconds
#     hours, seconds =  str((seconds // 3600) // len(times)).zfill(2), seconds % 3600
#     minutes, seconds = str(seconds // 60).zfill(2), str(seconds % 60).zfill(2)
#     average_morning = "{}:{}:{}".format(hours, minutes, seconds)
    
#     seconds = total_night_seconds
#     hours, seconds =  str((seconds // 3600) // len(times)).zfill(2), seconds % 3600
#     minutes, seconds = str(seconds // 60).zfill(2), str(seconds % 60).zfill(2)
#     average_night = "{}:{}:{}".format(hours, minutes, seconds)
    
#     return { \
#             'min_morning': min_morning, \
#             'average_morning': average_morning, \
#             'max_morning': max_morning, \
#             'min_night': min_night, \
#             'average_night': average_night, \
#             'max_night': max_night \
#            }


# In[ ]:


# circadian_rythm_data = get_average_circadian_rythm_time(circadian_rythms)
# circadian_rythm_data


# In[ ]:


# import requests
# microwave_stovetop = {'microwave': uq49_microwave_normalised, 'stovetop': uq49_stove_normalised}
# circadian_rythm_metrics = circadian_rythm_data
# circadian_rythm_test = {'morning': '05:28:00', 'night': '22:21:00'}
# data={'microwave_stovetop': microwave_stovetop, \
#       'circadian_rythm_metrics': circadian_rythm_metrics, \
#       'circadian_rythm_test': circadian_rythm_test, \
#      }

# r = requests.post('http://localhost:3001/data', json=data)
# print(r.json())
# print(r.status_code, '\n')
# pp.pprint(data)


# In[ ]:





# In[ ]:


# TODO: Export data from DB to csv???


# In[ ]:


# Nighttime Disturbances------------------------------------------------------------------------------------


# In[26]:


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


# In[27]:


# Get spikes (Watts greater than houses resting power use)...
# of less than 30 mins in duration between their circadian rhythm times.
def get_sleep_disturbances(data):
    sleep_disturbances = []
    circadian_rythms = []
    for household in data:
        curr_disturbances = get_peaks(household, 5, 1000, 1, 30, 60)
        try:
            circadian_rhythm = get_circadian_rythm(household)
#             print(circadian_rhythm)

            curr_disturbances = [peak for peak in curr_disturbances if compareTime(peak[0][11:], circadian_rhythm[0]) < 0                                  or compareTime(peak[0][11:], circadian_rhythm[1]) > 0]
            sleep_disturbances.append(curr_disturbances)
        except ZeroDivisionError:
            sleep_disturbances.append(-1)
#             print('div zero error')

    return [len(arr) if type(arr) != int else arr for arr in sleep_disturbances]


# In[41]:


# sleep_disturbances = get_sleep_disturbances(light_processed)
# sleep_disturbances


# # In[ ]:


# sleep_disturbances_normalised = []
# for val in sleep_disturbances:
#     if val == -1:
#         sleep_disturbances_normalised.append(-2)
#     else:
#         sleep_disturbances_normalised.append(normalise(val, min([x for x in sleep_disturbances if x != -1]), max(sleep_disturbances)))
# sleep_disturbances_normalised


# In[ ]:


# q = """SELECT * FROM "iotawatt" WHERE "device" = 'uq68' AND "sensor" = 'Powerpoints1' AND "time" >= '2020-09-01T00:00:00Z' AND "time" <= '2020-10-1T00:00:00Z'"""

# test_data = pd.DataFrame(client2.query(q, chunked=True))
# uq49_light_data


# In[ ]:


# uq49_light_data_processed = pre_process(uq49_light_data)
# uq49_light_data_processed


# In[ ]:


# uq49_sleep_disturbances = get_sleep_disturbances(uq49_light_data_processed)
# uq49_sleep_disturbances


# In[ ]:


# uq49_sleep_disturbances_normalised = normalise(uq49_sleep_disturbances[0], min(sleep_disturbances), max(sleep_disturbances))
# uq49_sleep_disturbances_normalised


# In[ ]:


# Work Schedule-------------------------------------------------------------------------------------------------------


# In[28]:


def get_work_schedules(data):
    lst = []
    for household in data:
        try:
            lst.append(get_work_schedule(household))
        except ZeroDivisionError:
#             print("zero div err")
            lst.append(0)
#             lst.append(('0', '0'))
            
    return lst


# In[ ]:


# work_schedules = get_work_schedules(light_processed)
# work_schedules


# In[29]:


def hoursBetween(first, second, overnight=False):
    a = datetime.datetime.strptime(first, '%H:%M:%S')
    b = datetime.datetime.strptime(second, '%H:%M:%S')
    if (overnight == True):
        temp = b
        b = a + datetime.timedelta(days=1)
        a = temp
    diff = b - a
    return int(diff.total_seconds() // 60)


# In[30]:


def get_work_durations(work_schedules):
    duration = []
    for tup in work_schedules:
        if (tup == 0):
            duration.append(-2)
        else:
            duration.append(hoursBetween(tup[0], tup[1]))
    return duration        


# In[ ]:


# work_durations = get_work_durations(work_schedules)
# work_durations


# # In[ ]:


# work_durations_normalised = []
# for val in work_durations:
#     if val == -2:
#         work_durations_normalised.append(-2)
#     else:
#         work_durations_normalised.append(normalise(val, min([x for x in work_durations if x != -2]), max(work_durations)))
# work_durations_normalised


# In[ ]:


# uq49_work_schedule = get_work_schedules(uq49_light_data_processed)
# uq49_work_schedule


# In[ ]:


# uq49_work_duration = get_work_duration(uq49_work_schedule)
# uq49_work_duration


# In[ ]:


# uq49_work_duration_normalised = normalise(uq49_work_duration[0], min(work_durations), max(work_durations))
# uq49_work_duration_normalised


# In[ ]:


# Microwave/Stovetop -------------------------------------------------------------------------------------------------


# In[ ]:


# print(microwave_peaks)
# print(stove_peaks)


# # In[ ]:


# microwave_normalised = []
# for val in microwave_peaks:
#     microwave_normalised.append(normalise(val, min(microwave_peaks), max(microwave_peaks)))
# microwave_normalised


# # In[ ]:


# stovetop_normalised = []
# for val in stove_peaks:
#     stovetop_normalised.append(normalise(val, min(stove_peaks), max(stove_peaks)))
# stovetop_normalised


# In[ ]:


# Sleep Duration ----------------------------------------------------------------------------------------------------


# In[ ]:


# sleep_schedules = get_all_circadian_rythm(light_processed)
# sleep_schedules


# In[31]:


def get_sleep_durations(sleep_schedules):
    duration = []
    for tup in sleep_schedules:
        if (tup == 0):
            duration.append(-2)
        else:
            duration.append(hoursBetween(tup[0], tup[1], True))
    return duration  


# In[ ]:


# sleep_durations = get_sleep_durations(sleep_schedules)
# sleep_durations


# In[ ]:


# sleep_durations_normalised = []
# for val in sleep_durations:
#     if val == -2:
#         sleep_durations_normalised.append(-2)
#     else:
#         sleep_durations_normalised.append(normalise(val, min([x for x in sleep_durations if x != -2]), max(sleep_durations)))
# sleep_durations_normalised


# In[183]:


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
# ages = { 
#         "uq10": 40,
#         "uq12": 38,
#         "uq23": 49,
#         "uq24": 53,
#         "uq26": 41,
#         "hfs01a": 35
#        }


# In[184]:


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

# stove = { 
#         "uq10": 0,
#         "uq12": 1,
#         "uq23": 1,
#         "uq24": 1,
#         "uq26": 1,
#         "hfs01a": 1
#         }


# In[185]:


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

# hotwater = { 
#         "uq10": 1,
#         "uq12": 1,
#         "uq23": 0,
#         "uq24": 1,
#         "uq26": 1,
#         "hfs01a": 1
#         }


# In[33]:





# In[81]:


# RANKING ---------------------------------------------------------------------------------------------------------


# In[59]:


# Takes in the variables of measuring stress/anxiety [-1,1] and returns a between [0,100] to be displayed
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


# In[197]:


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


# In[35]:


# Returns a ranked value between 0 and 100
# Got this wrong. Should take in a bunch of [-1,1] values are return a [0,100] value for the entire subject
def rank(oldValue):
    oldMin = -1
    oldMax = 1 
    newMin = 0
    newMax = 100 # The scale we want to show in the visualisations
    
    oldRange = (oldMax - oldMin)  
    newRange = (newMax - newMin)  
    newValue = (((oldValue - oldMin) * newRange) / oldRange) + newMin
    
    return int(newValue)


# In[ ]:


# uq49_sleep_disturbances_ranked = rank(uq49_sleep_disturbances_normalised)
# uq49_sleep_disturbances_ranked


# In[ ]:


# uq49_work_duration_ranked = rank(uq49_work_duration_normalised)
# uq49_work_duration_ranked


# In[ ]:


# stress_anxiety = stress_anxiety(uq49_sleep_disturbances_normalised, uq49_work_duration_normalised)
# stress_anxiety


# In[ ]:


# This is the stress/anxiety indexes of the 6 households that data was queried from.
# TODO: Extract more data (depth and breadth) and apply the same process to extract more meaningful values.
# TODO: Then, turn the ranks into a normalised value (with respect to the other households) between [0,100]
# stress_anxiety_all = stress_anxiety(sleep_disturbances_normalised, work_durations_normalised, sleep_durations_normalised, microwave_normalised, stovetop_normalised)


# In[36]:


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
        print('here')
        print(list(values.values()))
    return values


# In[ ]:


# result = normalise_and_rank(stress_anxiety_all, list(light_circuits.keys()))


# In[75]:





# In[ ]:





# In[67]:


def process_stress_anxiety():
    months = ['2020-11-01', '2020-12-01', '2020-12-31']
    
    final_result = {}
    
    for j in range(len(months)-1):
        print("--- Processing {} => {} ---".format(months[j], months[j+1]))
        
        # Microwave Data
        print("Getting microwave data")
        microwave_dataframes = []
        for i, home in enumerate(power_circuits):
            microwave_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(power_circuits[home])))
        
        # Stove Data
        print("Getting stove data")
        stove_dataframes = []
        for i, home in enumerate(stove_circuits):
            stove_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(stove_circuits[home])))           
        
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
            light_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(light_circuits[home])))
        
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
        sleep_schedules = get_all_circadian_rythm(light_processed)

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
            consumption_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(power_circuits[home])))
        
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


# In[52]:


# res = process_stress_anxiety()
# res


# In[70]:





# In[71]:


# {'uq10': 100, 'uq12': 16, 'uq23': 70, 'uq24': 45, 'uq26': 16, 'hfs01a': 25}


# In[ ]:





# In[44]:


# consumption_dataframes = []
# for i, home in enumerate(power_circuits):
#     consumption_dataframes.append(get_data_in_period(client2, "'2021-03-01T00:00:00Z'", "'2021-03-31T23:59:50Z'", "'{}'".format(home), "'Consumption'"))


# In[49]:


def get_consumption(data):
    total = 0
    for val in data:
        total += val
    return int((total / len(data)) * (60*60*24/10))


# In[45]:


# consumption_processed = pre_process(consumption_dataframes)
# consumption_processed


# In[73]:





# In[74]:





# In[46]:


def get_all_consumptions(data):
    consumptions = []
    for household in data:
        consumptions.append(get_consumption(household))
    return consumptions


# In[50]:


# consumptions = get_all_consumptions(consumption_processed)
# consumptions


# In[53]:


# consumption_normalised = []
# for val in consumptions:
#     if val == -2:
#         consumption_normalised.append(-2)
#     else:
#         consumption_normalised.append(normalise(val, min([x for x in consumptions if x != -2]), max(consumptions)))
# consumption_normalised


# In[ ]:





# In[ ]:





# In[69]:





# In[66]:





# In[ ]:





# In[ ]:





# In[72]:





# In[75]:


# TODO: Get the oven data, process it, count occurences of the oven being on for > 2 hours using get_peaks() 


# In[103]:


# oven_dataframes = []
# for i, home in enumerate(oven_circuits):
#     if (oven_circuits[home] != "none"):
#         oven_dataframes.append(get_data_in_period(client2, "'2021-03-01T00:00:00Z'", "'2021-03-31T23:59:50Z'", "'{}'".format(home), "'{}'".format(oven_circuits[home])))
#     else:
#         oven_dataframes.append(pd.DataFrame())
# oven_dataframes


# # In[104]:


# len(oven_dataframes)


# # In[107]:


# oven_processed = pre_process(oven_dataframes)


# In[133]:


def get_oven(data):
    res = get_peaks(data, 800, 10000, 120, 720, 0)
    if len(res) < 1:
        return -2
    else:
        return len(res)


# In[134]:


def get_all_ovens(data):
    ovens = []
    for household in data:
        if (household.empty):
            ovens.append(-2)
        else:
            ovens.append(get_oven(household))
    return ovens


# In[135]:


# ovens = get_all_ovens(oven_processed)
# ovens


# In[136]:


# oven_normalised = []
# for val in ovens:
#     if val == -2:
#         oven_normalised.append(-2)
#     else:
#         oven_normalised.append(normalise(val, min([x for x in ovens if x != -2]), max(ovens)))
# oven_normalised


# In[ ]:


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
            consumption_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(power_circuits[home])))
            
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
                oven_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(oven_circuits[home])))
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


# In[ ]:


# start = datetime.datetime.now()

# res2 = process_home_risk()

# end = datetime.datetime.now()
# print("Start: ", start)
# print("End: ", end)

# res2


# In[ ]:





# In[143]:


# Prostate/kidney disease
def health_insurance(ages, microwave, stovetop, sleep_disturbances, sleep_durations):
    
    c
    


# In[ ]:

    print("Prostate/Kidney Disease Ranks: \t\t\t", ranks)




# In[149]:


def process_health_insurance():
    months = ['2020-07-01', '2020-08-01', '2020-09-01'] #, '2020-10-01', '2020-11-01', '2020-12-01', '2020-12-31']
    
    final_result = {}
    
    for j in range(len(months)-1):
        print("--- Processing {} => {} ---".format(months[j], months[j+1]))
        
        # 1. Age
        print("1. Assessing age")

        # Microwave Data
        print("Getting microwave data")
        microwave_dataframes = []
        for i, home in enumerate(power_circuits):
            microwave_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(power_circuits[home])))
        
        # Stove Data
        print("Getting stove data")
        stove_dataframes = []
        for i, home in enumerate(stove_circuits):
            stove_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(stove_circuits[home])))           
        
        # Preprocessing 
        print("Processing microwave data")
        microwave_processed = pre_process(microwave_dataframes)
        print("Processing stove data")
        stove_processed = pre_process(stove_dataframes)

        del microwave_dataframes
        del stove_dataframes  
        
        # 2. Microwave vs Stovetop Use
        # TODO: Stove circuits currently uses a mixture of hotplate and oven circuits. Need to discuss this.
        print("2. Calculating microwave vs stovetop use")
        microwave_peaks, stove_peaks = microwave_stove_peaks(microwave_processed, stove_processed)
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
            light_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(light_circuits[home])))

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
        sleep_schedules = get_all_circadian_rythm(light_processed)

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


# In[150]:


# res3 = process_prostate_kidney_disease()
# res3


# In[ ]:





# In[162]:


def process_outlier_check():
    months = ['2020-10-01', '2020-11-01', '2020-12-01']     #, '2021-01-01', '2021-02-01', '2021-03-01']
    
    outliers = {}
    
    for j in range(len(months)-1):
        print("--- Processing {} => {} ---".format(months[j], months[j+1]))
        
        # Consumption Data
        print("Getting consumption data")
        consumption_dataframes = []
        for i, home in enumerate(power_circuits):
            consumption_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(power_circuits[home])))
        
        # Preprocessing
        print("Processing consumption data")
        consumption_processed = pre_process(consumption_dataframes)
        
        
        # 1. Outlier Checks
        print("1. Detecting outliers...")
        peaks = []
        for household in consumption_processed:
            peaks.append(get_peaks(household, 4000, 20000, 120, 720, 0))
        
        # Save outliers
        for i, key in enumerate(list(power_circuits.keys())):
            try:
                outliers[key] + peaks[i]
            except KeyError:
                outliers[key] = peaks[i]
        print("Current result: ", outliers)
        print()
    
    print("--- Outliers ---\n", outliers)        
    return outliers


# In[161]:


# res4 = process_outlier_check()
# res4


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[164]:


# TODO: Optimise pre_process(), expand breadth of households analysed, analyse a larger timeframe.


# In[ ]:



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
                aircon_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(aircon_circuits[home])))
        
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
        print("BOM: ", bom_data_max[j])
        print("USAGES: \n")
        for usage in usages:
            print(usage)
            if (usage != -100):
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
                aircon_dataframes.append(get_data_in_period(client2, "'{}T00:00:00Z'".format(months[j]), "'{}T00:00:00Z'".format(months[j+1]), "'{}'".format(home), "'{}'".format(aircon_circuits[home])))
        
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







if __name__ == '__main__':
    # start = datetime.datetime.now()

    # res = process_health_insurance()

    # end = datetime.datetime.now()

    # print()
    # print("Start: ", start)
    # print("End: ", end)

    # print("Time taken: {} minutes".format(end.time().minute - start.time().minute))

    # q = """SELECT * FROM "iotawatt" WHERE "device" = 'uq68' AND "sensor" = 'Powerpoints1' AND "time" >= '2020-09-25T00:00:00Z' AND "time" <= '2020-10-01T00:00:00Z'"""

    # test_data = pd.DataFrame(client2.query(q, chunked=True))
    # print("DATA: ", test_data[0][0])
        



    process_cold_sensitivity()









# %%
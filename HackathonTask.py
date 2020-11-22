import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
from haversine import haversine


def get_rider_weekdays(rider, orders):
    times = orders[orders['driverID'] == rider].pickupTime
    return np.array([t.weekday() for t in times])


def get_driver_days(orders):
    drivers_ids = list(set(orders.driverID))
    driver_days = defaultdict(lambda: [0, 6])
    for driver in drivers_ids:
        wds = get_rider_weekdays(driver, orders)
        c = collections.Counter(wds).most_common()
        if len(c) > 1:
            driver_days[driver] = [c[-2][0], c[-1][0]]
        if len(c) == 1:
            if c[0][0] == 0:
                driver_days[0] = 1
            elif c[0][0] == 6:
                driver_days[1] = 1
    return driver_days


def get_driver_cost(orders):
    driver_cost = defaultdict(lambda: [[0, 0] for i in range(7)])
    for row in orders.to_numpy():
        weekday = row[5].weekday()
        driver_cost[row[0]][weekday][0] += 1
        driver_cost[row[0]][weekday][1] += row[4]
    return driver_cost


def get_driver_mean(orders):
    drivers_ids = list(set(orders.driverID))
    driver_cost = get_driver_cost(orders)
    driver_mean = defaultdict(lambda: [0] * 7)
    for driver in drivers_ids:
        for i in range(7):
            if driver_cost[driver][i][0] == 0:
                continue
            driver_mean[driver][i] = driver_cost[driver][i][1] / driver_cost[driver][i][0]
    return driver_mean


def get_best_weekdays_for_away_drivers(drivers, orders):
    gdm = get_driver_mean(orders)
    away_drivers = drivers[drivers['away'] == 1].driverID.to_numpy()
    best_days = {}
    week_best_days = [0] * 7
    for d in away_drivers:
        best_days[d] = [[0, 1], np.argsort(np.array(gdm[d])), np.array(gdm[d])]
        week_best_days[best_days[d][1][0]] += 1
        week_best_days[best_days[d][1][1]] += 1

    n = (away_drivers.shape[0] * 2) // 7

    step = 0
    while min(week_best_days) < n - 2:
        step += 1
        change = None
        best_loss = 100000
        for i in range(7):
            if week_best_days[i] > n:
                for d in away_drivers:
                    if drivers[drivers.driverID == d]['class'].to_numpy()[0] != 1:
                        ind, args = best_days[d][0], best_days[d][1]
                        if i in ind:
                            for j in range(7):
                                if week_best_days[j] < n - 2 and j not in ind:
                                    loss = gdm[2][i] - gdm[2][j]
                                    if loss < best_loss:
                                        best_loss = loss
                                        change = (i, j, d)

        if change is None:
            break

        week_best_days[change[0]] -= 1
        week_best_days[change[1]] += 1

        if best_days[change[2]][0][0] == change[0]:
            best_days[change[2]][0][0] = change[1]
        else:
            best_days[change[2]][0][1] = change[1]

    return {d: best_days[d][0] for d in away_drivers}


def set_best_weekdays_for_away_drivers(drivers, orders):
    away_drivers = drivers[drivers['away'] == 1].driverID.to_numpy()
    bwfad = get_best_weekdays_for_away_drivers(drivers, orders)
    for d in away_drivers:
        drivers.loc[drivers.driverID == d, 'weekday1'] = min(bwfad[d][0], bwfad[d][1])
        drivers.loc[drivers.driverID == d, 'weekday2'] = max(bwfad[d][0], bwfad[d][1])


def clear_dataset(orders):
    orders.pickupTime = pd.to_datetime(orders.pickupTime)
    orders.dropoffTime = pd.to_datetime(orders.dropoffTime)
    orders = orders[orders["fare"] < 500]
    orders = orders[orders.dropoffTime is not pd.NaT and (orders.dropoffTime < pd.Timestamp('2020-11-09 00:00:00'))]
    orders[['rideRating']] = orders[['rideRating']].fillna(value=-1)
    return orders


def rider_class(rider, orders, thr1=0.28, thr2=0.17, thr3=0.23):
    grades = orders[orders['driverID'] == rider]['rideRating'].to_numpy()
    nrides = grades.shape[0]
    grades = [(grades == i).mean() for i in [-1, 1, 2, 3, 4, 5]]
    positive = grades[-1]
    # negative = sum(grades[1:-1])
    if nrides < 30:
        if positive > thr3:
            return 1
        return 2
    if positive > thr1:
        return 1
    if positive > thr2:
        return 2
    return 3


def get_long_dists(drivers):
    dists = np.zeros(drivers.shape[0], dtype='int')
    cx = orders.dropoff_lat.mean()
    cy = orders.dropoff_lon.mean()
    for i, row in drivers.iterrows():
        x = float(row['lat'])
        y = float(row['lon'])
        if haversine((x, y), (cx, cy)) > 10:
            dists[i] = 1
    return dists


def get_angle(c, p):
    if c[0] > p[0]:
        if c[1] > p[1]:
            return 1
        return 2
    if c[1] > p[1]:
        return 3
    return 4


def driver_features(drivers, orders):
    classes = [rider_class(drivers['driverID'][i], orders) for i in range(drivers.shape[0])]
    away = get_long_dists(drivers)
    drivers['away'] = away
    drivers['class'] = classes
    drivers['weekday1'] = -1
    drivers['weekday2'] = -1
    cx = orders.dropoff_lat.mean()
    cy = orders.dropoff_lon.mean()
    drivers['angle'] = [get_angle((cx, cy), (drivers['lat'][i], drivers['lon'][i])) for i in range(drivers.shape[0])]

    driver_days = get_driver_days(orders)
    drivers['current_weekday1'] = -1
    drivers['current_weekday2'] = -1
    for _, row in drivers.iterrows():
        row['current_weekday1'] = min(driver_days[row.driverID])
        row['current_weekday2'] = max(driver_days[row.driverID])

    return drivers


def get_free_weekday_slots(near_drivers):
    # sec - среднее количество заказов по дням недели
    sec = [4781.807692307692, 4919.192307692308, 4838.7307692307695, 5069.653846153846, 5582.0, 5475.185185185185, 4698.481481481482]
    free_slots = len(near_drivers) * 2
    aveps = np.sum(sec) / free_slots
    slots = [free_slots / 7] * 7

    for i in range(len(sec)):
        val = (sec[i] - np.mean(sec)) / aveps
        slots[i] -= val
        slots[i] = int(slots[i]) + 1

    return slots


def initial_distribution(near_drivers, driver_mean):
    res = [0] * 7
    distr = defaultdict(lambda: [0, 0])
    for _, row in near_drivers.iterrows():
        els = np.argsort(driver_mean[row.driverID])[:2]
        res[els[0]] += 1
        res[els[1]] += 1
        distr[row.driverID] = els
    return res, distr


def get_near_drivers_distribution(drivers, orders):
    near_drivers = drivers[drivers.away == 0]
    second_class = near_drivers[near_drivers['class'] == 2]
    gdm = get_driver_mean(orders)
    step = 0
    slots = get_free_weekday_slots(near_drivers)
    ptr, driver_distr = initial_distribution(near_drivers, gdm)
    while any([ptr[i] > slots[i] for i in range(7)]):
        step += 1
        change = None
        best_loss = 100000
        for i in range(7):
            if ptr[i] > slots[i]:
                for _, d in second_class.iterrows():
                    ind = driver_distr[d.driverID]
                    if i in ind:
                        for j in range(7):
                            if ptr[j] < slots[j] and j not in ind:
                                loss = gdm[2][i] - gdm[2][j]
                                if loss < best_loss:
                                    best_loss = loss
                                    change = (i, j, d.driverID)

        if change is None:
            break

        ptr[change[0]] -= 1
        ptr[change[1]] += 1
        ind = driver_distr[change[2]]

        if ind[0] == change[0]:
            ind[0] = change[1]
        else:
            ind[1] = change[1]
    return driver_distr


orders = clear_dataset(pd.read_csv('orders.csv'))
drivers = driver_features(pd.read_csv('drivers.csv'), orders)
set_best_weekdays_for_away_drivers(drivers, orders)

print(drivers.head())

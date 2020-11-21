import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def clear_dataset(orders):
    orders.dropoffTime = pd.to_datetime(orders.dropoffTime)
    orders = orders[orders["fare"] < 500]
    orders = orders[orders.dropoffTime is not pd.NaT and (orders.dropoffTime < pd.Timestamp('2020-11-09 00:00:00'))]
    orders[['rideRating']] = orders[['rideRating']].fillna(value=-1)
    return orders


def rider_class(rider, thr1=0.28, thr2=0.17, thr3=0.23):
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


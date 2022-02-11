#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

from asyncio import SubprocessTransport
import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = '/Users/chinmay/Documents/GitHub/cs181-s22-homeworks/HW1/data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
        
    basis = np.ones(xx.shape) # Bias term
    if part == 'a':
        for j in range(1, 6):
            basis = np.vstack((basis, pow(xx, j)))
    elif part == 'b':
        for j in range(1960, 2011, 5):
            basis = np.vstack((basis, (np.exp((-1 * (xx-j)**2) / 25))))
    elif part == 'c':
        for j in range(1, 6):
            basis = np.vstack((basis, np.cos(xx / j)))
    elif part == 'd':
        for j in range (1, 26):
            basis = np.vstack((basis, np.cos(xx / j)))
    basis = basis.T
    return basis

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

def get_rss(Yhat, Y):
    return np.sum(np.power((Y - Yhat), 2))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)

# Plot and report sum of squared error for each basis
for i, letter in enumerate(['a', 'b', 'c', 'd']):
    plt.subplot(2, 2, i+1)

    X_basis = make_basis(years, part=letter, is_years=True)
    weights = find_weights(X_basis, Y)
    Yhat = np.dot(X_basis, weights)
    rss = get_rss(Yhat, Y)

    grid_X_basis = make_basis(grid_years, part=letter, is_years=True)
    grid_Yhat = np.dot(grid_X_basis, weights)

    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel("Year")
    plt.ylabel("# Republicans in Senate")
    plt.title("Basis \"" + letter + "\" \n" +
        "Residual sum of squares: " + str(round(rss, 2)))
plt.suptitle("Year vs. Number of Republicans in the Senate")
plt.tight_layout()
plt.show()

# Sunspots vs. Republicans.

grid_sunspots = np.linspace(np.amin(sunspot_counts), np.amax(sunspot_counts), 200)

for i, letter in enumerate(['a', 'c', 'd']):
    plt.subplot(2, 2, i+1)

    X_basis = make_basis(sunspot_counts[0:13], part=letter, is_years=False)
    weights = find_weights(X_basis, Y[0:13])
    Yhat = np.dot(X_basis, weights)
    rss = get_rss(Yhat, Y[0:13])

    grid_X_basis = make_basis(grid_sunspots, part=letter, is_years=False)
    grid_Yhat = np.dot(grid_X_basis, weights)

    plt.plot(sunspot_counts, republican_counts, 'o', grid_sunspots, grid_Yhat, '-')
    plt.xlabel("# Sunspots")
    plt.ylabel("# Republicans in Senate")
    plt.title("Basis \"" + letter + "\" \n" +
        "Residual sum of squares: " + str(round(rss, 2)))
plt.suptitle("Number of Sunspots vs. Number of Republicans in the Senate")
plt.tight_layout()
plt.show()
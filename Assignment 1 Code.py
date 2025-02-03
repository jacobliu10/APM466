import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import newton
from scipy.interpolate import interp1d

class Bond:
    def __init__(self, identifier, price_history, maturity_date, coupon_rate, payment_periods):
        self.identifier = identifier
        self.price_history = price_history
        self.maturity_date = maturity_date
        self.coupon_rate = coupon_rate
        self.payment_periods = payment_periods

def compute_accrued_interest(coupon_rate, days_since_last_coupon, days_in_coupon_period):
    """Calculate accrued interest for a bond."""
    return (coupon_rate / 2) * (days_since_last_coupon / days_in_coupon_period)

def compute_present_value(bond_instance, observation_day, ytm):
    """Calculate the dirty price of a bond (clean price + accrued interest)."""
    clean_price = bond_instance.price_history[observation_day]
    accrued_interest = compute_accrued_interest(bond_instance.coupon_rate, 183, 183)  # Placeholder for actual calculation
    dirty_price = clean_price + accrued_interest
    return dirty_price

def sum_present_value(face_value, coupon_amount, num_periods, discount_rate, year_fractions):
    """Calculate the present value of a bond's cash flows using discrete compounding."""
    accumulated_value = 0
    for period_index in range(1, num_periods + 1):
        # Discrete compounding: (1 + r/n)^(n*t)
        discount_factor = 1 / (1 + discount_rate / 2) ** (2 * year_fractions[period_index - 1])
        accumulated_value += coupon_amount * discount_factor
    # Add the present value of the face value
    accumulated_value += face_value * (1 / (1 + discount_rate / 2) ** (2 * year_fractions[-1]))
    return accumulated_value

def find_ytm(bond_instance, observation_day, year_fractions):
    """Calculate the yield to maturity (YTM) using discrete compounding."""
    def pv_diff(ytm):
        pv = sum_present_value(100, bond_instance.coupon_rate / 2, bond_instance.payment_periods, ytm, year_fractions)
        dirty_price = compute_present_value(bond_instance, observation_day, ytm)
        return pv - dirty_price
    
    try:
        ytm = newton(pv_diff, 0.03)  # Initial guess of 3%
        return ytm * 100  # Convert to percentage
    except RuntimeError:
        print(f"YTM calculation failed for bond {bond_instance.identifier} on day {observation_day}")
        return np.nan  # Return NaN if YTM calculation fails

def calculate_daily_ytm(bond_objects_list):
    """Calculate YTM for each bond on each day."""
    daily_ytm = {}
    daily_year_fraction = {}

    for day_index in range(10):
        ytm_values = []
        year_fractions = []

        for bond_index, bond_instance in enumerate(bond_objects_list):
            # Calculate time to maturity in years
            year_fraction = (bond_instance.maturity_date - date_mapping[day_index]).days / 365
            year_fractions.append(year_fraction)
            # Calculate YTM
            ytm = find_ytm(bond_instance, day_index, year_fractions)
            ytm_values.append(ytm)

        daily_ytm[day_index] = ytm_values
        daily_year_fraction[day_index] = year_fractions

    return daily_ytm, daily_year_fraction

# Load data
file_location = r"C:\Users\jacob\Downloads\APM466 Assignment 1.csv"
bond_dataset = pd.read_csv(file_location)

# Print column names to inspect them
print(bond_dataset.columns)

selected_identifiers = ['CA135087K940', 'CA135087L518', 'CA135087L930', 'CA135087M847', 
                        'CA135087N837', 'CA135087P576', 'CA135087Q491', 'CA135087Q988', 'CA135087R895', 'CA135087S471']

filtered_bond_data = bond_dataset[bond_dataset["ISIN"].isin(selected_identifiers)]

bond_objects_list = []
for idx, row in filtered_bond_data.iterrows():
    price_over_days = [row[col_index] for col_index in range(4, 14)]  # Adjust the range if necessary
    maturity_datetime = datetime.strptime(row["Maturity Date"], "%m/%d/%Y")
    annual_coupon = float(row["Coupon"].replace("%", ""))
    payment_count = int((maturity_datetime - datetime.now()).days / 182.5)
    
    bond_instance = Bond(row["ISIN"], price_over_days, maturity_datetime, annual_coupon, payment_count)
    bond_objects_list.append(bond_instance)

date_mapping = {
    0: datetime(2025, 1, 6),
    1: datetime(2025, 1, 7),
    2: datetime(2025, 1, 8),
    3: datetime(2025, 1, 9),
    4: datetime(2025, 1, 10),
    5: datetime(2025, 1, 13),
    6: datetime(2025, 1, 14),
    7: datetime(2025, 1, 15),
    8: datetime(2025, 1, 16),
    9: datetime(2025, 1, 17)
}

daily_ytm, daily_year_fraction = calculate_daily_ytm(bond_objects_list)

# Plotting
date_labels = ['Jan 6', 'Jan 7', 'Jan 8', 'Jan 9', 'Jan 10', 'Jan 13', 'Jan 14', 'Jan 15', 'Jan 16', 'Jan 17']
plt.xlabel('Time to Maturity (Years)')
plt.ylabel('Yield to Maturity (%)')
plt.title('Canadian Government Bonds 5 Year Yield Curves')

for day_index in range(10):
    year_fractions = daily_year_fraction[day_index]
    ytm_values = daily_ytm[day_index]
    interp_func = interp1d(year_fractions, ytm_values, kind='cubic', fill_value="extrapolate")
    x_new = np.linspace(0, 5, 100)  # Plot from x = 1 to x = 5
    y_new = np.clip(interp_func(x_new), 0, None)  # Ensure interpolated values are non-negative
    plt.plot(x_new, y_new, label=date_labels[day_index])

# Set x-axis ticks and labels
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['1', '2', '3', '4', '5'])
plt.xlim(1, 5)  # Ensure the x-axis ranges from 1 to 5
plt.ylim(2, 3.2)

plt.legend(loc=1, prop={'size': 6})
plt.show()

def calculate_spot_rates(bond_objects_list, daily_ytm, daily_year_fraction):
    daily_spot_rates = {}

    for day_index in range(10):
        spot_rates = []
        year_fractions = daily_year_fraction[day_index]
        ytm_values = daily_ytm[day_index]

        # Sort bonds by maturity
        sorted_indices = np.argsort(year_fractions)
        sorted_year_fractions = np.array(year_fractions)[sorted_indices]
        sorted_ytm_values = np.array(ytm_values)[sorted_indices]

        # Initialize spot rates
        spot_rates = [sorted_ytm_values[0] / 100]  # First spot rate is the YTM of the shortest bond

        for i in range(1, len(sorted_year_fractions)):
            bond_instance = bond_objects_list[sorted_indices[i]]
            coupon = bond_instance.coupon_rate / 2
            face_value = 100
            price = compute_present_value(bond_instance, day_index, sorted_ytm_values[i] / 100)

            # Calculate the present value of the coupon payments
            pv_coupons = 0
            for j in range(i):
                pv_coupons += coupon * np.exp(-spot_rates[j] * sorted_year_fractions[j])

            # Solve for the spot rate
            def spot_rate_diff(r):
                return pv_coupons + (coupon + face_value) * np.exp(-r * sorted_year_fractions[i]) - price

            try:
                spot_rate = newton(spot_rate_diff, spot_rates[-1])  # Use the last spot rate as initial guess
                spot_rates.append(spot_rate)
            except RuntimeError:
                print(f"Spot rate calculation failed for bond {bond_instance.identifier} on day {day_index}")
                spot_rates.append(np.nan)  # Return NaN if spot rate calculation fails

        daily_spot_rates[day_index] = spot_rates

    return daily_spot_rates, sorted_year_fractions

# Calculate spot rates
daily_spot_rates, sorted_year_fractions = calculate_spot_rates(bond_objects_list, daily_ytm, daily_year_fraction)

# Plotting
plt.xlabel('Time to Maturity (Years)')
plt.ylabel('Spot Rate (%)')
plt.title('Canadian Government Bonds 5 Year Spot Rate Curves')

for day_index in range(10):
    spot_rates = daily_spot_rates[day_index]
    interp_func = interp1d(sorted_year_fractions, spot_rates, kind='cubic', fill_value="extrapolate")
    x_new = np.linspace(1, 5, 100)  # Plot from x = 1 to x = 5
    y_new = np.clip(interp_func(x_new), 0, None)  # Ensure interpolated values are non-negative
    plt.plot(x_new, y_new * 100, label=date_labels[day_index])  # Convert to percentage

# Set x-axis ticks and labels
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['1', '2', '3', '4', '5'])
plt.xlim(1, 5)  # Ensure the x-axis ranges from 1 to 5

plt.legend(loc=1, prop={'size': 6})
plt.show()

def calculate_forward_rates(daily_spot_rates, sorted_year_fractions):
    daily_forward_rates = {}

    for day_index in range(10):
        spot_rates = daily_spot_rates[day_index]
        forward_rates = []

        # Calculate 1-year forward rates for terms 2-5 years
        for n in range(1, 5):  # n = 1, 2, 3, 4 (for 1yr-1yr, 1yr-2yr, 1yr-3yr, 1yr-4yr)
            t = 2  # Starting point is always 1 year
            t_plus_n = t + 2*n  # Ending point   is t + 2n years since n is 1 period

            # Find the spot rates for t and t+n
            S_t = spot_rates[2*t - 1]  # Spot rate for t years (index t-1 because Python is 0-based)
            S_t_plus_n = spot_rates[t_plus_n - 1]  # Spot rate for t+n years

            # Calculate the forward rate using the formula
            forward_rate = ((1 + S_t_plus_n) ** t_plus_n / (1 + S_t) ** t) ** (1 / n) - 1
            forward_rates.append(forward_rate/2)

        daily_forward_rates[day_index] = forward_rates

    return daily_forward_rates

daily_forward_rates = calculate_forward_rates(daily_spot_rates, sorted_year_fractions)

# Plotting the forward rates
plt.xlabel('Forward Term (Years)')
plt.ylabel('Forward Rate (%)')
plt.title('1-Year Forward Rates (2-5 Years)')

# Define the forward terms (1yr-1yr, 1yr-2yr, 1yr-3yr, 1yr-4yr)
forward_terms = [1, 2, 3, 4]  # Corresponding to 1yr-1yr, 1yr-2yr, 1yr-3yr, 1yr-4yr

# Plot forward rates for each day
for day_index in range(10):
    forward_rates = daily_forward_rates[day_index]
    plt.plot(forward_terms, [rate * 100 for rate in forward_rates], label=date_labels[day_index])

# Set x-axis ticks and labels
plt.xticks(ticks=forward_terms, labels=['1yr-1yr', '1yr-2yr', '1yr-3yr', '1yr-4yr'])
plt.xlim(1, 4)  # Ensure the x-axis ranges from 1 to 4

# Add legend
plt.legend(loc='upper right', prop={'size': 6})

# Show the plot
plt.show()

from scipy.linalg import eig

# Part 5: Calculate covariance matrices for daily log-returns of yield and forward rates

# Part 5: Calculate covariance matrices for daily log-returns of yield and forward rates

def calculate_log_returns(rates):
    """
    Calculate daily log-returns for a given set of rates.
    """
    log_returns = np.diff(np.log(rates), axis=0)
    return log_returns

# Extract 5-year yield rates (1-year, 2-year, 3-year, 4-year, 5-year)
yield_rates = np.array([daily_ytm[day_index][1:6] for day_index in range(10)])  # Shape: (10, 5)

# Extract 5 forward rates (1yr-1yr, 1yr-2yr, 1yr-3yr, 1yr-4yr, 1yr-5yr)
forward_rates = np.array([daily_forward_rates[day_index] for day_index in range(10)])  # Shape: (10, 5)

# Calculate daily log-returns for yield rates
yield_log_returns = calculate_log_returns(yield_rates)

# Calculate daily log-returns for forward rates
forward_log_returns = calculate_log_returns(forward_rates)

# Calculate covariance matrices
yield_cov_matrix = np.cov(yield_log_returns, rowvar=False)  # Covariance matrix for yield log-returns
forward_cov_matrix = np.cov(forward_log_returns, rowvar=False)  # Covariance matrix for forward log-returns

print("Covariance matrix for yield log-returns (5x5):")
print(yield_cov_matrix)

print("\nCovariance matrix for forward log-returns (5x5):")
print(forward_cov_matrix)

# Part 6: Calculate eigenvalues and eigenvectors of the covariance matrices

def calculate_eigenvalues_and_eigenvectors(cov_matrix):
    """
    Calculate eigenvalues and eigenvectors of a covariance matrix.
    """
    eigenvalues, eigenvectors = eig(cov_matrix)
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors

# Calculate eigenvalues and eigenvectors for yield covariance matrix
yield_eigenvalues, yield_eigenvectors = calculate_eigenvalues_and_eigenvectors(yield_cov_matrix)

# Calculate eigenvalues and eigenvectors for forward covariance matrix
forward_eigenvalues, forward_eigenvectors = calculate_eigenvalues_and_eigenvectors(forward_cov_matrix)

print("\nEigenvalues for yield covariance matrix:")
print(yield_eigenvalues)

print("\nEigenvectors for yield covariance matrix:")
print(yield_eigenvectors)

print("\nEigenvalues for forward covariance matrix:")
print(forward_eigenvalues)

print("\nEigenvectors for forward covariance matrix:")
print(forward_eigenvectors)

# Interpretation of the first eigenvalue and eigenvector
def interpret_first_eigenvalue_and_eigenvector(eigenvalues, eigenvectors):
    """
    Interpret the first eigenvalue and its associated eigenvector.
    """
    first_eigenvalue = eigenvalues[0]
    first_eigenvector = eigenvectors[:, 0]
    print(f"\nFirst eigenvalue: {first_eigenvalue}")
    print(f"First eigenvector: {first_eigenvector}")
    print("Interpretation: The first eigenvector represents the direction of maximum variance in the data.")
    print("The first eigenvalue indicates the magnitude of this variance.")

# Interpret the first eigenvalue and eigenvector for yield rates
print("\nInterpretation for yield rates:")
interpret_first_eigenvalue_and_eigenvector(yield_eigenvalues, yield_eigenvectors)

# Interpret the first eigenvalue and eigenvector for forward rates
print("\nInterpretation for forward rates:")
interpret_first_eigenvalue_and_eigenvector(forward_eigenvalues, forward_eigenvectors)

import matplotlib.pyplot as plt
import numpy as np

# Office Measurement data
# heights = [23.71, 23.23, 23.61, 23.37, 23.61, 23.61, 23.61, 23.61, 23.61, 23.49, 
#          23.49, 23.61, 23.29, 23.32, 23.41, 23.32, 23.41, 23.41, 23.32, 23.32,
#          23.32, 23.32, 23.50, 23.50, 23.50, 23.50, 23.26, 23.26, 23.26, 23.50,
#          23.26, 23.50, 23.41, 23.29, 23.62, 23.62, 23.29, 23.41, 23.41, 23.41,
#          23.53, 23.41]

# widths = [18.26, 18.34, 18.42, 18.32, 18.42, 18.42, 18.42, 18.42, 18.42, 18.32,
#          18.32, 18.42, 18.41, 18.50, 18.50, 18.50, 18.50, 18.50, 18.50, 18.50,
#          18.50, 18.50, 18.34, 18.34, 18.34, 18.34, 18.33, 18.33, 18.33, 18.34,
#          18.33, 18.34, 18.26, 18.25, 18.35, 18.35, 18.25, 18.34, 18.26, 18.26,
#          18.44, 18.26]

# West Facing Measurement data
# heights = [24.59, 24.76, 24.76, 24.99, 24.44, 24.73, 24.99, 24.45, 24.30, 24.34, 24.34, 23.91, 23.91, 23.86, 24.10]

# widths = [18.99, 19.26, 19.34, 19.43, 19.01, 19.25, 19.43, 19.59, 19.34, 19.58, 19.58, 19.39, 19.67, 19.37, 19.30]

# East Facing Measurement data
heights = [25.09, 24.88, 23.64, 24.56, 25.45, 24.14, 24.78, 23.76, 25.29, 25.90, 24.96, 25.30, 24.08, 24.14, 23.98]

widths = [20.11, 20.26, 19.69, 19.94, 20.70, 18.69, 19.62, 18.91, 19.92, 20.26, 19.80, 20.18, 18.99, 18.50, 18.88]

# Real measurements and calculations
real_height = 24.35
real_width = 18.7

height_mean = np.mean(heights)
width_mean = np.mean(widths)
height_error_cm = height_mean - real_height
width_error_cm = width_mean - real_width

# Convert to inches (1 inch = 2.54 cm)
height_error_inch = height_error_cm / 2.54
width_error_inch = width_error_cm / 2.54

# Target error boundaries (1/16 inch = 0.15875 cm)
target_error = (1/16) * 2.54
height_upper = real_height + target_error
height_lower = real_height - target_error
width_upper = real_width + target_error
width_lower = real_width - target_error

# Calculate standard deviations
height_std = np.std(heights)
width_std = np.std(widths)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('East Facing Measurements', fontsize=16)

# Height distribution
ax1.hist(heights, bins=15, alpha=0.7, color='blue')
ax1.axvline(real_height, color='red', linestyle='-', label=f'Real Height ({real_height}cm)')
ax1.axvline(height_mean, color='green', linestyle='-', label=f'Mean ({height_mean:.2f}cm)')
ax1.axvline(height_upper, color='orange', linestyle='--', label=f'Target (±1/16")')
ax1.axvline(height_lower, color='orange', linestyle='--')
ax1.set_title('Height Distribution')
ax1.set_xlabel('Height (cm)')
ax1.set_ylabel('Frequency')
ax1.legend()

# Add height error annotation
ax1.annotate(
   f'Error: {height_error_cm:.3f}cm ({height_error_inch:.3f}in)\n'
   f'Std Dev: {height_std:.3f}cm',
   xy=(height_mean, ax1.get_ylim()[1]),
   xytext=(height_mean + 0.1, ax1.get_ylim()[1] * 0.9),
   bbox=dict(facecolor='white', alpha=0.8)
)

# Width distribution
ax2.hist(widths, bins=15, alpha=0.7, color='blue')
ax2.axvline(real_width, color='red', linestyle='-', label=f'Real Width ({real_width}cm)')
ax2.axvline(width_mean, color='green', linestyle='-', label=f'Mean ({width_mean:.2f}cm)')
ax2.axvline(width_upper, color='orange', linestyle='--', label=f'Target (±1/16")')
ax2.axvline(width_lower, color='orange', linestyle='--')
ax2.set_title('Width Distribution')
ax2.set_xlabel('Width (cm)')
ax2.set_ylabel('Frequency')
ax2.legend()

# Add width error annotation 
ax2.annotate(
   f'Error: {width_error_cm:.3f}cm ({width_error_inch:.3f}in)\n'
   f'Std Dev: {width_std:.3f}cm',
   xy=(width_mean, ax2.get_ylim()[1]),
   xytext=(width_mean + 0.1, ax2.get_ylim()[1] * 0.9),
   bbox=dict(facecolor='white', alpha=0.8)
)

plt.tight_layout()
plt.show()

# Print statistics
print("\nMeasurement Statistics:")
print(f"Height:")
print(f"Mean: {height_mean:.3f}cm")
print(f"Std Dev: {height_std:.3f}cm")
print(f"Error: {height_error_cm:.3f}cm ({height_error_inch:.3f}in)")
print(f"Target Range: {height_lower:.3f} to {height_upper:.3f}cm")

print(f"\nWidth:")
print(f"Mean: {width_mean:.3f}cm")
print(f"Std Dev: {width_std:.3f}cm") 
print(f"Error: {width_error_cm:.3f}cm ({width_error_inch:.3f}in)")
print(f"Target Range: {width_lower:.3f} to {width_upper:.3f}cm")
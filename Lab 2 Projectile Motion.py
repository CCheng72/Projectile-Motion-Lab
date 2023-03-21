#!/usr/bin/env python
# coding: utf-8

# # Projectile Motion Laboratory #
# 
# You will a produce a scientific lab report in this Jupyter Notebook. Use markdown cells to display text and equations, and code cells to do calculations and make plots. Fill in the blanks and change anything you'd like.
# 
# ### Zhihao Cheng ###
# 
# #### Instructor: Robert Carey ####
# 
# #### Lab TF: Celia Mercovich ####

# In[2]:


get_ipython().run_line_magic('pylab', 'notebook')

from scipy.optimize import curve_fit #this is the function you'll use for fitting


# ## Abstract ##
# 
# This experiment is to measure the range of a small steel ball after going through a projectile motion. Evantually we need to find if the measured ranges in 2 trials, one with a horizontal launch angle and another one with a positive launch angle, are consistent with the estimated ranges, including errors, using the horizontal and vertical launch velocity component. Results: the measured range has a small difference compared to the estimate range with errors in the horizontal launch, but there the angled launch the difference is much larger, which could result from measurement errors such as incorrect readings during the experiment. Other than error factors, the overall experimental design is appropriate for both trials; the data is reasonable; and the calculations for velocities and laucnh angles is accurate as well. 

# ## Procedure ##
# 
# To reach our final results, here are the procedures for our experiment: 
# 
# - Set up a projectile motion track using a plastic tube. A small steel ball is chosen to be the launch object. 
# - Two trials are set for the projectile motion experiment: the first one we launch the steel ball with a horizontal launch angle, that is, the vertical component of the launch velocity is zero; the second one we launch the steel ball with a launch angle of ... 
# - The origin of the launch is when at the instant the steel ball comes out the tube. We then use a high-speed cellphone camera (30 frames per second) to determine the x and y coordinates in each frame. Therefore, we have the x and y coordinates from 0 to 0.53s (about 15-16 frames) for the horizontal launch trial and coordinates from 0 to 0.93s (27 frames) for the angled launch trial. However, due to the slow motion captures of pictures, the observed time lapse is 4 times shorter than the original frame speed. Therefore, when importing time coordinates into bothe trials' graphs we need to divide the time by 4 in order to get the calculations correctly.
# - After recording those x and y coordinates for both trials using a cell phone camera frame by frame, we measure the x and y measurement errors and will use them in the final approach towards calculating the propagated errors of $ \Delta R $.
# - Plots of the horizontal and vertical components of the launch velocity will be drawn at last. Partical derivatives in the error propagation for $ \Delta R $, $\Delta v_o$, and $\Delta \theta$ will be calculated.

# ## Analysis ##
# 
# 4 graphs will be drawn separately: x & y coordinates for the horizontal launch and x & y coordinates for the angled launch. Each data point will have an error of $0.0045m$/$4.5mm$ due to our filming. The error bars will be on all 4 graphs but the bar sizes on each graph differ as the limit of x and y axes are all different. 
# After putting all data points (position coordinates) on our graphs, we need to make a least square fitting line to get optimal values. The lines' slopes will be used to determine horizontal and vertical velocity components, final estimated launch velocity, and drag/air resistance measurements for 2 trials. As there will be errors, we also need the error propagation methods to find the final error for our launch velocity, angle, and height so that the estimated $\Delta R$ can be calculated. Also one thing to note that due to our filming there might be a lack of coordinate data for some frames. The error of heights is approximately $1mm$, which is $0.001m$.

# In[9]:


#import data
t1, x1, y1, x1_error, y1_error = loadtxt('Horizontal Launch Angle Data.txt', delimiter = '\t', skiprows = 1, unpack = True)
t2, x2, y2, x2_error, y2_error = loadtxt('Launch_with_angle Data.txt', delimiter = '\t', skiprows = 2, unpack = True)

for i in range(len(t1)):
    t1[i] = t1[i]/4
for j in range(len(t2)):
    t2[j] = t2[j]/4

print(t1)
print(type(t1[1]))


# In[161]:


#my measured values

#initial height
h1 = 0.523
h2 = 0.584
#error in height
dh1 = 0.001
dh2 = 0.001
#range
R_m1 = 0.772
R_m2 = 0.837
#error in range
dR_m1 = 0.001
dR_m2 = 0.001


# In[6]:


def linear(x, m, b):
    y = m*x + b
    return y
def quadratic(x, a, b, c):
    y = a*x**2 + b*x + c
    return y


def fit_horizontal_x():
    figure()
    grid()
    errorbar(t1, x1, yerr = x1_error, label = 'data', marker = 'o', linestyle = 'None')
    popt, pcov = curve_fit(linear, t1, x1, sigma = x1_error)

    m1 = popt[0]
    m1_err = np.sqrt(diag(pcov)[0])
    print('The fitted slope is: ', round(m1,3), '+-', round(m1_err,3))


    b1 = popt[1]
    b1_err = np.sqrt(diag(pcov)[1])
    print('The fitted intercept is: ', round(b1,2), '+-', round(b1_err,2))

    plot(t1, linear(t1, *popt), label = 'fit')

    legend()
    title("x coordinates for the horizontal launch ($v_ox$ is the slope)")
    xlabel("Time (s)")
    ylabel("Horizontal displacement (m)")
    
fit_horizontal_x()

def fit_horizontal_y():
    figure()
    grid()
    errorbar(t1, y1, yerr = y1_error, label = 'data', marker = 'o', linestyle = 'None')
    p0 = [0.01, 0.02, 0.03]
    popt, pcov = curve_fit(quadratic, t1, y1, p0 = p0, sigma = y1_error)
    plot(t1, quadratic(t1, *popt), label = 'fit')
    
    a1 = popt[0]
    a1_err = np.sqrt(diag(pcov)[0])
    print('The fitted coefficient for t^2 (a) is: ', round(a1,3), '+-', round(a1_err,3))


    b1 = popt[1]
    b1_err = np.sqrt(diag(pcov)[1])
    print('The fitted coefficient for t (b/v_oy) is: ', round(b1,3), '+-', round(b1_err,3))

    
    c1 = popt[2]
    c1_err = np.sqrt(diag(pcov)[2])
    print('The fitted coefficient c is: ', round(c1,3), '+-', round(c1_err,3))

    
    legend()
    title("y coordinates for the horizontal launch ($v_oy$ is the slope)")
    xlabel("Time (s)")
    ylabel("Vertical displacement (m)")
    
fit_horizontal_y()


# # Horizontal Launch
# 
# For the horizontal launch, due to the construction of our projectile motion settings and parts we use, we fail to make the steel ball launch flatly. Therefore, there is a small initial launch angle of approximately $10.49 ^\circ$. The initial height for the launch is approximately $52.3cm/0.523m$. Due to the limits of the y-axis, the error bars of x coordinates for the horizontal launch are too small to be seen. After doing the line fittings for both x and y coordinates, we have the slopes for first function and the coefficient of t for second function, which are the approximate vertical and horizontal velocity components: $$v_ox = 2.075 \pm 0.023 m/s$$ $$v_oy = 0.278 \pm 0.037 m/s$$
# In order to calculate the final launch velocity for our horizontal launch, we need the Pythagoras theorem:
# $$v_o = \sqrt{v_ox^2 + v_oy^2}$$
# and error propagation methods to calculate the uncertainties:
# $$|\eth f| = \sqrt{(\partial f/\partial x)^2 \eth x^2 + (\partial f/\partial y)^2 \eth y^2}$$
# In this case:
# $$|\eth v_o| = \sqrt{(\partial v_o/\partial v_ox)^2 \eth v_ox^2 + (\partial v_o/\partial v_oy)^2 \eth v_oy^2}$$
# As we have the equation of the Pythagoras Theorem, we can take the partial derivatives for both $v_ox$ and $v_oy$: 
# $$\frac{\partial v_o}{\partial v_ox} = \frac{1}{2}(v_ox^2 + v_oy^2)^\frac{-1}{2}(2v_ox)$$ 
# $$\frac{\partial v_o}{\partial v_oy} = \frac{1}{2}(v_ox^2 + v_oy^2)^\frac{-1}{2}(2v_oy)$$
# Therefore, we can calculate both partial derivatives and thus get the final uncertantity of v_o:
# $$\frac{\partial v_o}{\partial v_ox} = \frac{1}{2}(2.075^2 + 0.278^2)^\frac{-1}{2}(2\times2.075) \approx 0.991$$
# $$\frac{\partial v_o}{\partial v_oy} = \frac{1}{2}(0.519^2 + 0.069^2)^\frac{-1}{2}(2\times0.278) \approx 0.133$$
# Finally, we substitute all variables to the equation of our $|\eth v_o|$ and $v_o$:
# $$|\eth v_o| = \sqrt{(0.991)^2 0.023^2 + (0.133)^2 0.037^2} \approx 0.023 m/s$$
# $$v_o = \sqrt{2.075^2 + 0.278^2} \approx 2.094 m/s$$
# Hence, we can have the conclusion that the launch velocity for our horizontal launch $v_o$ is approximately $2.094 \pm 0.023 m/s$.
# 

# From here, we can calculate the angle $\theta$:
# $$\tan(\theta) = \frac{v_oy}{v_ox}$$
# Therefore, we have the formulae:
# $$\theta = \tan^{-1}(\frac{v_oy}{v_ox})$$
# $$|\eth \theta| = \sqrt{(\frac{\partial \theta}{\partial v_ox})^2 \eth v_ox^2 + (\frac{\partial \theta}{\partial v_oy})^2 \eth v_oy^2}$$
# $$\frac{\partial \theta}{\partial v_ox} = \frac{1}{1+(\frac{v_oy}{v_ox})^2} \cdot v_oy \cdot -(v_ox)^{-2}$$
# $$\frac{\partial \theta}{\partial v_oy} = \frac{1}{1+(\frac{v_oy}{v_ox})^2} \cdot \frac{1}{v_ox}$$
# After substituing all values into the equations above: we can get both $\theta$ and $d\theta$:
# $$\frac{\partial \theta}{\partial v_ox} = \frac{1}{1+(\frac{0.278}{2.075})^2} \cdot 0.278 \cdot -(2.075)^{-2} \approx -0.063$$
# $$\frac{\partial \theta}{\partial v_oy} = \frac{1}{1+(\frac{0.278}{2.075})^2} \cdot \frac{1}{2.075} \approx 0.473$$
# $$\theta = \tan^{-1}(\frac{0.278}{2.075}) \approx 7.631^\circ$$
# $$|\eth \theta| = d\theta = \sqrt{(-0.063)^2 2.075^2 + 0.473^2 0.278^2} \approx 0.185^\circ$$
# Finally, we have the launch angle and its associated error: $$\theta_h = 7.631 \pm 0.185^\circ$$
# From here we can see that the launch angle conforms to our initial claim that the launch is not completely horizontal.
# 

# In[163]:


#also use a quadratic fit to check your x data for drag, or another method of your choice


# In[164]:


def quadratic(x, a, b, c):
    y = a*x**2 + b*x + c
    return y

v_o = 0.594
v_o_err = 0.010
force = 0.25*v_o**2


def fit_horizontalx_drag():
    figure()
    grid()
    errorbar(t1, x1, yerr = x1_error, label = 'data', marker = 'o', linestyle = 'None')
    popt, pcov = curve_fit(quadratic, t1, x1, sigma = x1_error)
    
    a1 = popt[0]
    a1_err = np.sqrt(diag(pcov)[0])
    print('The fitted coefficient of t^2 (a) is: ', round(a1,3), '+-', round(a1_err,3))

    b1 = popt[1]
    b1_err = np.sqrt(diag(pcov)[1])
    print('The fitted coefficient of t (b) is: ', round(b1,3), '+-', round(b1_err,3))
    
    c1 = popt[2]
    c1_err = np.sqrt(diag(pcov)[2])
    print('The coefficient c is: ', round(c1,3), '+-', round(c1_err,3))
    
    plot(t1, quadratic(t1, *popt), label = 'fit')
    
    legend()
    title("Air Resistance Measurement in the Horizontal Launch")
    xlabel("Time (s)")
    ylabel("x coordinates for the horizontal launch (m)")
    
fit_horizontalx_drag()
print(f'The air drag/resistance force (x direction) is {force}N')


# ## Air Resistance Measurement in the Horizontal Launch
# 
# For the air resistance measurement, we use the quadratic fit. We plot the time as the x-axis and x coordinates as y-axis into the quadratic fit. The final fitted line has a value a, b, and c:
# $$a = -0.583 \pm 0.646$$
# $$b = 2.153 \pm 0.09$$
# $$c = 0.004 \pm 0.003$$
# As we check the coefficient of $t^2$, which is a, we find that the air resistance value is no larger than its associated error. Therefore, we can say that the air drag here is insignificant. The calculated air drag is about $0.088N$.

# In[165]:


def linear(x, m, b):
    y = m*x + b
    return y
def quadratic(x, a, b, c):
    y = a*x**2 + b*x + c
    return y

'''
for this fit you may need a p0 for the in order for the function to converge on optimal values. input an array of the form:
p0 = [a, b, c] where you have visually estimated these guesses from a plot of your data
'''

def fit_angle_x():
    figure()
    grid()
    errorbar(t2, x2, yerr = x2_error, label = 'data', marker = 'o', linestyle = 'None')
    popt, pcov = curve_fit(linear, t2, x2, sigma = x2_error)

    m2 = popt[0]
    m2_err = np.sqrt(diag(pcov)[0])
    print('The fitted slope is : ', round(m2,3), '+-', round(m2_err,3))


    b2 = popt[1]
    b2_err = np.sqrt(diag(pcov)[1])
    print('The fitted intercept is : ', round(b2,2), '+-', round(b2_err,2))


    plot(t2, linear(t2, *popt), label = 'fit')

    legend()
    title("x coordinates for the angled launch ($v_ox$ is the slope)")
    xlabel("Time (s)")
    ylabel("Horizontal displacement (m)")


fit_angle_x()


def fit_angle_y():
    figure()
    grid()
    errorbar(t2, y2, yerr = y2_error, label = 'data', marker = 'o', linestyle = 'None')
    p0 = [0.01, 0.02, 0.03]
    popt, pcov = curve_fit(quadratic, t2, y2, p0 = p0, sigma = y2_error)
    plot(t2, quadratic(t2, *popt), label = 'fit')
    
    m2 = popt[0]
    m2_err = np.sqrt(diag(pcov)[0])
    print('The fitted coefficient of t^2 (a)  is : ', round(m2,3), '+-', round(m2_err,3))


    b2 = popt[1]
    b2_err = np.sqrt(diag(pcov)[1])
    print('The fitted coefficient of t (b/v_oy) is : ', round(b2,3), '+-', round(b2_err,3))

    
    c2 = popt[2]
    c2_err = np.sqrt(diag(pcov)[2])
    print('The fitted coefficient c is: ', round(c2,3), '+-', round(c2_err,3))
    
    legend()
    title("y coordinates for the angled launch")
    xlabel("Time (s)")
    ylabel("Vertical displacement (m)")
    
fit_angle_y()


# # Vertical Launch
# 
# For the vertical launch, the initial height for the launch is approximately $58.4cm/0.584m$. Due to the same problem in the horizontal launch (a too small limit of the y-axis), the error bars of x coordinates for our angled launch are too small to be seen on the graph as well. After doing the line fittings for both x and y coordinates, we also have the approximate vertical and horizontal velocity components: 
# $$v_ox = 1.704 \pm 0.017 m/s$$ $$v_oy = 0.770 \pm 0.023 m/s$$. 
# Similar to the horizontal launch approach, we use the Pythagoras theorem and the error propagation methods to calculate the final launch velocity: 
# $$\frac{\partial v_o}{\partial v_ox} = \frac{1}{2}(v_ox^2 + v_oy^2)^\frac{-1}{2}(2v_ox) = \frac{1}{2}(1.704^2 + 0.770^2)^\frac{-1}{2}(2\times1.704) \approx 0.911$$
# $$\frac{\partial v_o}{\partial v_oy} = \frac{1}{2}(v_ox^2 + v_oy^2)^\frac{-1}{2}(2v_oy) = \frac{1}{2}(1.704^2 + 0.770^2)^\frac{-1}{2}(2\times0.770) \approx 0.412$$
# And after substituting all variables into the our $|\eth v_o|$ and $v_o$ functions:
# $$|\eth v_o| = \sqrt{(\frac{\partial v_o}{\partial v_ox})^2 \eth v_ox^2 + (\frac{\partial v_o}{\partial v_oy})^2 \eth v_oy^2} = \sqrt{(0.911)^2 0.017^2 + (0.412)^2 0.023^2} \approx 0.018 m/s$$
# $$v_o = \sqrt{v_ox^2 + v_oy^2} = \sqrt{1.704^2 + 0.770^2} \approx 1.870 m/s$$
# Eventually, the result is that the launch velocity for our angled launch $v_o$ is approximately $1.870 \pm 0.018 m/s$.

# The launch angle $\theta_a$ and its associated error is calculated below:
# $$\frac{\partial \theta}{\partial v_ox} = \frac{1}{1+(\frac{v_oy}{v_ox})^2} \cdot v_oy \cdot -(v_ox)^{-2} = \frac{1}{1+(\frac{0.77}{1.704})^2} \cdot 0.77 \cdot -(1.704)^{-2} \approx -0.220$$
# $$\frac{\partial \theta}{\partial v_oy} = \frac{1}{1+(\frac{v_oy}{v_ox})^2} \cdot \frac{1}{v_ox} = \frac{1}{1+(\frac{0.77}{1.704})^2} \cdot \frac{1}{1.704} \approx 0.487$$
# $$\theta = \tan^{-1}(\frac{v_oy}{v_ox}) = \tan^{-1}(\frac{0.77}{1.704}) \approx 24.317^\circ$$
# $$|\eth \theta| = d\theta = \sqrt{(\frac{\partial \theta}{\partial v_ox})^2 \eth v_ox^2 + (\frac{\partial \theta}{\partial v_oy})^2 \eth v_oy^2} = \sqrt{(-0.220)^2 0.017^2 + (0.487)^2 0.023^2} \approx 0.012^\circ$$
# $$\theta_a = 24.317 \pm 0.012^\circ$$

# This expression is error propogation for R. Compute each term yourself:
# 
# $$ \Delta R = \sqrt{ \Delta v_0^2 \left(\frac{dR}{dv_0}\right)^2 + \Delta \theta^2 \left(\frac{dR}{d\theta}\right)^2 + \Delta h^2 \left(\frac{dR}{dh}\right)^2 } $$

# In[166]:


#compute values for the horizontal launch, and check if their corresponding errors are appropriate

h1 = 0.523 #initial height
dh1 = 0.001 #error in height

theta1 = math.radians(7.631) #initial angle
dtheta1 = math.radians(0.185) #error in angle

v_01 = 2.094 #initial velocity, find with v_0x and v_0y
dv1 = 0.023  #error in velocity

g = 9.8

R_m1 = 0.772
dR_m1 = 0.001

#error propogation
R1 = ((v_01**2) * (cos(theta1)/g)) * (sin(theta1) + sqrt(sin(theta1)**2 + 2*h1*g/v_01**2))
R1 = round(R1,3)

dR1_dv01 = 2 * v_01 * cos(theta1)/g * (sin(theta1) + sqrt(sin(theta1)**2 + 2*h1*g/v_01**2)) + (2 * h1 * cos(theta1)/g) * (sin(theta1)**2 + 2*h1*g/v_01**2)**(-1/2)
dR1_dtheta1 = (v_01**2 * sin(theta1)/g) * (sin(theta1) + sqrt(sin(theta1)**2 + 2*h1*g/v_01**2)) + (v_01*cos(theta1))**2/g + (v_01**2*sin(theta1)*cos(theta1)**2/g) * (sin(theta1)**2 + 2*h1*g/v_01**2)**(-1/2)
dR1_dh1 = ((v_01**2) * (cos(theta1)/g)) * ((1/2)*(sin(theta1)**2 + 2*h1*g/(v_01**2))**(-1/2) * 2*g/v_01**2)

dR1 = sqrt(dv1**2*dR1_dv01**2 + dtheta1**2*dR1_dtheta1**2 + dh1**2*dR1_dh1**2)
dR1 = round(dR1,3)

print(dR1_dv01)
print(dR1_dtheta1)
print(dR1_dh1)

print(f'Final estimated range: {R1}m, the measured range: {R_m1} +-{dR_m1}m')
print(f'The calculated error of our estimated range: {dR1}m')
print(f'Final result: R = {R1} +- {dR1}m')


# In[167]:


#compute values for the angled launch, and check if their corresponding errors are appropriate

h2 = 0.584 #initial height
dh2 = 0.001 #error in height

theta2 = math.radians(24.317) #initial angle
dtheta2 = math.radians(0.012) #error in angle

v_02 = 1.87 #initial velocity, find with v_0x and v_0y
dv2 = 0.018 #error in velocity

g = 9.8

R_m2 = 0.837
dR_m2 = 0.001

#error propogation
R2 = v_02**2 * cos(theta2)/g * (sin(theta2) + sqrt(sin(theta2)**2 + 2*h2*g/v_02**2))
R2 = round(R2,3)

dR2_dv02 = 2 * v_02 * cos(theta2)/g * (sin(theta2) + sqrt(sin(theta2)**2 + 2*h2*g/(v_02**2))) + (2 * h2 * cos(theta2)/g) * (sin(theta2)**2 + 2*h2*g/v_02**2)**(-1/2) 
dR2_dtheta2 = (v_02**2 * sin(theta2)/g) * (sin(theta2) + sqrt(sin(theta2)**2 + 2*h2*g/v_02**2)) + (v_02*cos(theta2))**2/g + (v_02**2*sin(theta2)*cos(theta2)**2/g) * (sin(theta2)**2 + 2*h2*g/v_02**2)**(-1/2)
dR2_dh2 = ((v_01**2) * (cos(theta1)/g)) * ((1/2)*(sin(theta2)**2 + 2*h2*g/(v_02**2))**(-1/2) * 2*g/v_02**2)

dR2 = sqrt((dv2**2)*(dR2_dv02**2) + (dtheta2**2)*(dR2_dtheta2**2) + (dh2**2)*(dR2_dh2**2))
dR2 = round(dR2,3)

print(dR2_dv02)
print(dR2_dtheta2)
print(dR2_dh2)

print(f'Final estimated range: {R2}m, the measured range: {R_m2} +-{dR_m2}m')
print(f'The calculated error of our estimated range: {dR2}m')
print(f'Final result: R = {R2} +- {dR2}m')


# ## Results ##
# 
# After careful calculations, we have successfully found the estimated range along with error propagations:<br>
# <b>Horizontal launch</b>:<br>
# Estimated range: $0.739 \pm 0.018m$<br>
# Measured range: $0.772m \pm 0.001m$<br>
# <br>
# <b>Angled launch</b>:<br>
# Estimated range: $0.737 \pm 0.015m$<br>
# Measured rangeL $0.837 \pm 0.001m$<br>
# <br>
# From the data we can see that both of our measurements fail to be within the estimated range's errors. For the horizontal launch, the measured range, despite not within the errors, looks reasonable as there is only a small difference, which could be caused by observation imprecision. However, the angled launch measured range is about 0.1m larger than the estimated one, which is already 10cm. The difference is indeed not random and could be due to the incorrect use and readings for the ruler since there's no data mistake.
# In conclusion, except from possible error of ruler readings, the overall data collections, important parameters' calculations, and final range comparisons are rational and the experiment is successful as a whole. 

# In[169]:


def horizontal_path():
    figure()
    grid()
    plot(x1, y1, label = 'fit')
    
    legend()
    title("The predicted path for the horizontal launch")
    xlabel("x (m)")
    ylabel("y (m)")


def angled_path():
    figure()
    grid()
    plot(x2, y2, label = 'fit')
    
    legend()
    title("The predicted path for the angled launch")
    xlabel("x (m)")
    ylabel("y (m)")
    
horizontal_path()
angled_path()


# The predicted paths for both trials are shown above.

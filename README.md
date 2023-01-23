# What is it?

`covid-modeling` is a repository built with the purpose of fetching latest data, processing and modeling Covid-19 for reference and decision support. It is designed to be simple yet concrete so sufficient information can be retrieved. Users can modify criterias parameters to their desire, which will affect how processing and modeling steps will be done. Since creator is from Vietnam, a little more in-depth analysis is carried out for Vietnam.

# Methodology

Many groups of scientists have development complex models to predict the pattern of virus infection. Some of which are too complicated for normal people to understand, so a simple SVID model was built and applied in this repository.

A ordinary differential equation system is built to estimate the spread of Covid-19 virus base on the following information.

The population will be broken down into 4 groups (S+V+I+D=N):
* Susceptible (**S**): The main population which is susceptible to the disease without any protection **(person)**.
* Vaccinated (**V**): The population which has been vaccinated or recovers from the disease and has protection against the disease to some degree **(person)**.
* Infectious (**I**): The infected population, whom can infect the non-infected population **(person)**.
* Dead (**D**): The population which passes away due to the disease (disease-induced mortality) **(person)**.
* Affected population (**N**): The total population which is affected by the disease **(person)**.

<ins>The ODE (ordinary differential equation) model:</ins>
  
```math
\begin{align} 
&\frac{dS}{dt} = \frac{-\beta SI}{N} - (\alpha+\mu)S + \alpha_0V +\pi p \\
&\frac{dV}{dt} = \frac{-\beta_vVI}{N} -(\alpha_0 + \mu)V + \theta I + \alpha S + \pi(1-p) \\
&\frac{dI}{dt} = -(\mu +\gamma +\theta)I + \frac{(\beta S + \beta_V V)I}{N} \\
&\frac{dD}{dt} = -\mu D + \gamma I
\end{align} 
```

<ins>With initial conditions as follows:</ins>

```math
\begin{align} 
&S_0 \ge 0 \\
&V_0 \ge 0 \\
&I_0 \ge 0 \\
&D_0 \ge0
\end{align} 
```

<ins>And model parameter as follows:</ins>

$\pi$: recruitment rate $(person/time)$ <br>
$p$: recruitment non-vaccinated percent *(%)* <br>
$\alpha$: vaccination rate $(time^{-1})$ <br>
$\alpha_0$: disminissing rate of vaccine effectiveness $(time^{-1})$ <br>
$\beta$: transmission rate $(person/time)$ <br>
$\mu$: natural death rate $(time^{-1})$ <br>
$\gamma$: Covid-19 death rate $(time^{-1})$ <br>
$\theta$: post-infected immunity rate $(time^{-1})$ <br>
$\beta_v = \beta(1-\eta)$: vaccinated tranmission rate $(person/time)$ <br>

By solving the ODE system we can get the stability points:

```math
\begin{align}
&\frac{dS}{dt} = 0 \\
&\frac{dV}{dt} = 0 \\
&\frac{dI}{dt} = 0 \\
&\frac{dD}{dt} = 0
\end{align}
```

[The stability of typical equilibria of smooth ODEs is determined by the sign of real part of eigenvalues of the Jacobian matrix. An equilibrium is asymptotically stable if all eigenvalues have negative real parts; it is unstable if at least one eigenvalue has positive real part.](http://www.scholarpedia.org/article/Equilibrium#:~:text=Jacobian%20Matrix,-The%20stability%20of&text=where%20all%20derivatives%20are%20evaluated,eigenvalue%20has%20positive%20real%20part.)

```math
J=D_xf = f_x = \frac{\partial f_i}{\partial x_j} =
\begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \ldots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \ldots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_n}{\partial x_1} & \frac{\partial f_n}{\partial x_2} & \ldots & \frac{\partial f_n}{\partial x_n} 
\end{pmatrix}
```

From there, we can possibly have two answers:

* Disease free equilibrium (DFE): The point at which the population is disease-free,where as:

```math
\begin{align} 
&S = \frac{\alpha_0\pi + p\mu\pi}{\alpha\mu + \alpha_0\mu +\mu^2} \\
&V = \frac{\alpha\pi + (1-p)\mu\pi}{\alpha\mu + \alpha_0\mu + \mu^2} \\
&I=0 \\
&D=0 \\
&R_0 = \frac{\alpha\beta_v + \alpha_0\beta + p\beta\mu + \beta_v\mu}{(\gamma + \mu + \theta)(\alpha + \alpha_0 + \mu) + p\beta_v\mu} \lt 1
\end{align} 
```

* Endemic Equilibrium (EE): The point at which the disease is at equilibrium point (the disease is not totally eradicated and remains in the population).

```math
\begin{align} 
&S^*\geq 0 \\
&V^*\geq 0 \\
&I^*\geq 0 \\
&D^*\geq 0 \\
&R_0 = \frac{\alpha\beta_v + \alpha_0\beta + p\beta\mu + \beta_v\mu}{(\gamma + \mu + \theta)(\alpha + \alpha_0 + \mu)+p\beta_v\mu} \gt 1
\end{align} 
```

*where: $R_0$ is The Basic Reproduction Number that describes the transmissability or contagiousness of an infectious disease. If $R_0<1$, the disease will eventually die out and vice versa.

# Modeling

Since model parameters is time depedent and susceptible to external forces, it seems more appropriate to use forecasted parameters for model predictions instead of constant values. There are currently 2 method used to estimate the model parameters: *SARIMA* and *regression*.

The model will show users the current state of of the disease, future prediction for a set period of time and other information such as: daily case infected, daily recovery case, etc.<br>
E.g:<br>

<img src=https://user-images.githubusercontent.com/85442695/214021769-e1cd7254-e3b7-4df8-9ead-61e633e85655.png alt="example_1" width=75% height=75%><br>
<img src=https://user-images.githubusercontent.com/85442695/214021796-03832f64-8a54-49e6-95c9-2d19cfaa080b.png alt="example_2" width=75% height=75%><br>

# In-depth guideline
`TBA`
# Disclaimer
This is a personal project made with :sparkling_heart from Kha Nguyen Minh. It is used for acadamic and reference purposes. Creator have no influence and takes no responsibility for decisions made by agencies using this repository. 
# References
Data sources:
- [Our World in Data](https://github.com/owid/covid-19-data) - Worldwide Covid-19 database used in the model are mainly gathered from here.
- [VNCDC](https://ncov.vncdc.gov.vn/) - Vietnam Covid-19 database used in the model are mainly gathered from here.
- [VN Ministry of Health](https://tiemchungcovid19.gov.vn/portal) - Vietnam vaccination & distribution database.
- [Wikipedia](https://en.wikipedia.org/wiki/COVID-19_vaccination_in_Vietnam#cite_note-102) - Vietnam vaccine import data
- [CSSE at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19) - databases are used for comparison. 
- [UN Department of Social and Affairs](https://population.un.org/wpp/Download/Standard/CSV/) - census databases.
- [Vietnam General Statistics Office](https://www.gso.gov.vn) - Vietnam census database.
- [WHO](https://covid19.who.int/info) - vaccination metadata
- [Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/download-data) - Environment temperature data
- [Undata](http://data.un.org/Data.aspx?d=CLINO&f=ElementCode%3A16) - Wind speed data
- 
Model references:
`TBA`

SARIMA model references:
`TBA`

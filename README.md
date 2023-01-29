# What is it?

`covid-modeling` is a repository built with the purpose of fetching latest data, processing and modeling Covid-19 for reference and decision support. It is designed to be simple yet concrete so sufficient information can be retrieved. Users can modify criterias parameters to their desire, which will affect how processing and modeling steps will be done. Since creator is from Vietnam, a little more in-depth analysis is carried out for Vietnam.

`covid-modeling` is built as a local machine application, it uses browser UI and `Dash` to launch. Most of the working processes will happen on local machine without interactions with a server, except for downloading initial databases. The purpose of this is for faster calculation since the application will depend only on the local machine and remove the necessary of data transferring between user and server, which can sometime become a bottleneck for the application.

![Alt Text](https://media.giphy.com/media/4ocNpojqqNpa9ESB9i/giphy.gif)

# How to use?

There are 2 ways to use this repository:
- Use the given notebooks in the repository to work directly with if you want more control over parameters, the methodology is the same as the web application. Put the notebooks in a same folder and run it step by step (Collection>Preparation>Modeling), the all process results will be returned in `database` directory.
- Use the web application by running `main.py` for simple user interaction, it will cache the latest server databases to your local machine (which might take a while to create caches). After that, users can try to gather latest data (server is only updated weekly), change how to process raw data & modeling.

# Installation

After copy the repository to your local machine, it is recommended to create a new environment to work with and install all required packages from `requirement.txt`.

- For Conda distribution users:

Create new environment:
```
conda create --name <env_name>
```
Activate environment:
```
conda activate <env_name>
```
Download required packages:
```
pip install -r requirement.txt
```
- For new Python user: [Download Python 3.9.13](https://www.python.org/downloads/release/python-3913/)

Make new directory (for repository storage):
```
mkdir <repository_directory>
```
Change to directory:
```
cd <repository_directory>
```
Create new environment:
```
python -m venv <env_name>
```
Activate environment:

If iOS: `source <env_name>/bin/activate`, if Window: `.<env_name>\Scripts\activate`

Download required packages:
```
pip install -r requirement.txt
```
- Application start up:

After copy the repository and installation environment, we activate the environment and run the main script `python main.py` from a command prompt within the repository, copy the URL pop up in the command prompt to a browser to start the application.

# Limitation

Due to `background_callback` of `Dash` is incompatible with conversion step to executive file, it's currently not possible to run the application directly as an executable file. Therefore, unfamiliar users have to set up their own Python environment for the application to work. If anyone can help me set up `PyInstaller` or similar modules for Dash app, feel free to contact me.

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

# Data modeling

Since model parameters is time depedent and susceptible to external forces, it seems more appropriate to use forecasted parameters for model predictions instead of constant values. There are currently 2 method used to estimate the model parameters: *SARIMA* and *regression*.

The model will show users the current state of of the disease, future prediction for a set period of time and other information such as: daily case infected, daily recovery case, etc.<br>
E.g:<br>

<img src=https://user-images.githubusercontent.com/85442695/214021769-e1cd7254-e3b7-4df8-9ead-61e633e85655.png alt="example_1" width=75% height=75%><br>
<img src=https://user-images.githubusercontent.com/85442695/214021796-03832f64-8a54-49e6-95c9-2d19cfaa080b.png alt="example_2" width=75% height=75%><br>

# Disclaimer

This is a personal project made with :sparkling_heart: from Kha Nguyen Minh. It is used for acadamic and reference purposes. Creator have no influence and takes no responsibility for decisions made by agencies using this repository. 

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

Model methodology references: The repository was inspired by some of the references shown below.
- [Compartmental models in epidemiology](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#cite_note-Bailey1975-13) - Commonly used models in epidemic
- [The basic reproduction number of the new coronavirus pandemic with mortality for India, the Syrian Arab Republic, the United States, Yemen, China, France, Nigeria and Russia with different rate of cases](https://www.sciencedirect.com/science/article/pii/S221339842030186X) - Covid-19 SIRD model mathematical proof, with a good explaination on how to calculate reproduction rate from eigenvalue of Jacobian matrix.
- [Reproduction numbers of infectious disease models](https://www.sciencedirect.com/science/article/pii/S2468042717300209) - Various endemic models with calculated reproduction rates
- [Model the transmission dynamics of COVID-19 propagation with public health intervention](https://www.sciencedirect.com/science/article/pii/S2590037420300339) - an article on a complex model that shows the connections between exposed population, incubation period, quarantine population, etc. The article also explains how to calculate EE and DFE points.  
- [Mathematical assessment of the impact of non-pharmaceutical interventions on curtailing the 2019 novel Coronavirus](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7252217/) - A good article for understanding the model parameters meaning and calculate EE, DFE.
- [A Mathematical Model of COVID-19 with Vaccination and Treatment](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8421179/#B41) - This article show the effect of population change and vaccinations on infection rate, also some good reference parameter values to start.
- [Dynamic behaviors of a modified SIR model with nonlinear incidence and recovery rates](https://www.sciencedirect.com/science/article/pii/S1110016821000260#f0010) - this modified SIR model also include the effect of natural death rate to the model.
- [Optimal Control applied to SIRD model of COVID 19](https://arxiv.org/pdf/2109.01457.pdf) - SIRD model with a good mathematic proof of calculating reproduction rate and EE point.
- [A model and predictions for COVID-19 considering population behavior and vaccination](https://www.nature.com/articles/s41598-021-91514-7#data-availability) - A good simple model with effect of population and vaccination on infection rate, as well as consider recovered cases as vaccinated cases.
- [A systems approach to biology](https://vcp.med.harvard.edu/papers/SB200-3.pdf) - A lecture on system stability, how to use Jacobian matrix and calculate eigenvalue
- [Lecture on system stability](https://www.youtube.com/watch?v=NmntYoB1uJg&ab_channel=MITOpenCourseWare) - A good small course on system stability
- [Modelling Epidemic](https://jvanderw.une.edu.au/L5_ModellingEpidemics1.pdf) - A short lecture on some important endemic definitions

Model parameters: Some references on how to calculate model parameter of ODE systems from real|processed database
- [Modified SIRD Model for COVID-19 Spread Prediction for Northern and Southern States of India](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8120454/)
- [COVID-19 Lifecycle: Predictive Modelling of States in India](https://journals.sagepub.com/doi/pdf/10.1177/0972150920934642)
- [Estimating and Simulating a SIRD Model of COVID-19 for Many Countries, States, and Cities](https://pdfs.semanticscholar.org/d8bc/31e8f03e39f1f0c534d225f831f7d0bd05ab.pdf)
- [Fast estimation of time-varying infectious disease transmission rates](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008124) - Good article on how to calculate time depenent parameters.
- [Advanced epidemic models](https://mysite.science.uottawa.ca/rsmith43/MAT3395/AdvancedEpidemic.pdf) - Learn to understand 2 form of infection rates.

Python model references:
- [Mathematical Modelling: Modelling the Spread of Diseases with SIRD Model](https://www.analyticsvidhya.com/blog/2021/08/mathematical-modelling-modelling-the-spread-of-diseases-with-sird-model/)
- [COVID-19 data with SIR model](https://www.kaggle.com/code/lisphilar/covid-19-data-with-sir-model/notebook)
- [Modeling and Control of a Campus Outbreak of Coronavirus COVID-19](https://jckantor.github.io/CBE30338/03.09-COVID-19.html)
- [Analysis of a COVID-19 compartmental model: a mathematical and computational approach](https://www.aimspress.com/article/doi/10.3934/mbe.2021396?viewType=HTML)

SARIMA model references:
- [Autoregression Models for Time Series Forecasting With Python](https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/)
- [ARIMA Model – Complete Guide to Time Series Forecasting in Python](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)
- [A Gentle Introduction to SARIMA for Time Series Forecasting in Python](https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/)
- [Comprehensive Guide To Time Series Analysis Using ARIMA](https://analyticsindiamag.com/comprehensive-guide-to-time-series-analysis-using-arima/)

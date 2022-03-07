# PACKAGES
require(deSolve)
require(reshape2)
require(ggplot2)

library(deSolve)
library(reshape2)
library(ggplot2)
# INPUT

# Set up an empty contact matrix with rows for each age group and columns for each age group
contact_matrix <- matrix(0,nrow=3,ncol=3)
# Fill in the contract matrix
contact_matrix[1,1] = 7     # daily number of contacts that children make with each other
contact_matrix[1,2] = 5     # daily number of contacts that children make with adults
contact_matrix[1,3] = 1     # daily number of contacts that children make with the elderly
contact_matrix[2,1] = 2     # daily number of contacts that adults make with children
contact_matrix[2,2] = 9     # daily number of contacts that adults make with each other
contact_matrix[2,3] = 1     # daily number of contacts that adults make with the elderly
contact_matrix[3,1] = 1     # daily number of contacts that elderly people make with children
contact_matrix[3,2] = 3     # daily number of contacts that elderly people make with adults
contact_matrix[3,3] = 2     # daily number of contacts that elderly people make with each other
# The contact_matrix now looks exactly like the one in the etivity instructions. We add this matrix as a parameter below.

# Parameters
# initial b 0.05
parameters <- c(b = 0.0060,     # the probability of infection per contact is 5%
                contact_matrix = contact_matrix,   # the age-specific average number of daily contacts (defined above)
                gamma = 1/5)  # the rate of recovery is 1/5 per day

# Run simulation for 3 months
#initially, "by = 0.1" but to compare with the google dataset 
#I set "by = 1" to have the same timestep
#we obtain the same result as when "by" was set to 0.1

times <- seq(from = 0, to = 90, by = 1)

# MODEL FUNCTION
sir_age_model <- function(time, state, parameters) {  
  
  with(as.list(parameters), {
    
    n_agegroups <- 3                                 # number of age groups
    S <- state[1:n_agegroups]                        # assign to S the first 3 numbers in the initial_state_values vector
    I <- state[(n_agegroups+1):(2*n_agegroups)]      # assign to I numbers 4 to 6 in the initial_state_values vector
    R <- state[(2*n_agegroups+1):(3*n_agegroups)]    # assign to R numbers 7 to 9 in the initial_state_values vector
    
    N <- S+I+R     # people in S, I and R are added separately by age group, so N is also a vector of length 3
    
    # Defining the force of infection
    
    # Force of infection acting on susceptible individuals
    lambda <- b * contact_matrix %*% as.matrix(I/N)
    # %*% is used to multiply matrices in R
    # the lambda vector contains the forces of infection for children, adults and the elderly (length 3)
    
    # The differential equations
    # Rate of change in children:
    #dS <- -lambda^(2) * S
    dS <- -lambda*S
    #dS <- -lambda^3 * S
    #dS <- (log(lambda))*S 
    
    dI <- -dS - gamma * I
    #dI <- log(lambda*S)  - gamma * I
    #dI <- lambda^3 * S - gamma * I
    
    
    #dR <- gamma * I
    dR <- gamma * I
    
    
    # Output
    return(list(c(dS, dI, dR))) 
  })
}

#########################################
#Here we study the time period 1st July - 1st October 2021
#From Our World in Data:
#Fully vaccinated %, 1st of July: 38,2%
#Fully vaccinated %, 1st of October: 59,1%
#Fully vaccinated %, 15th of August (half of the period): 52,7%
#--> vaccination tr�s rapide au d�but
#le % monte vite � >50%. 
#Faire une moyenne des taux de vaccinations ...
#...sur un �chantillon des jours

#Pour trouver le vaccination coverage for each age group: nb fully vaccinated by age group / total number of people in the age group

#Repartition between age groups:
#nombre de fully vaccinated people by age groupe : https://www.covid19.admin.ch/en/vaccination/persons?ageGroupClass=vaccStrategy&demoSum=total&vaccPersonsRel=abs
#0-15 ; 16-64 ; >65

#purcentage of population for each age group obtained with https://www.bfs.admin.ch/bfs/fr/home/statistiques/population/effectif-evolution/age-etat-civil-nationalite.assetdetail.18264546.html
#(moyenne entre sexes)
#pas exactement la m�me tranche d'�ge (20-65 au lieu de 18-65 dans le mod�le, et 16-65 dans les stats vaccination)
#pop suisse totale: 8670300 (https://www.bfs.admin.ch/bfs/fr/home/statistiques/population.html)

#au 31.12.2020: +65 ans : 18,8% de la population -> ~1630016,4 personnes
#20-64 ans : 61,3% de la population -> ~5314893,9 personnes
#<20 ans : 19,9% de la population -> ~1725389,7 personnes

#donc: vaccine coverage (nb fully vacc / nb total age group)
#pas les m�mes chiffres pour chaque date. J'ai commenc� avec le 15 ao�t, la moiti� de la p�riode �tudi�e.
#04.07
#elder: 1317088/1630016,4
#adults : 2104756/5314893,9
#children : 640/1725389,7

#15.08
#elder: 1373914/1630016,4 = 84,39%
#adults : 3172399/5314893,9 = 59,69%
#children : 21988/1725389,7 = 1,27%

#03.10
#elder: 1416361/1630016,4 =
#adults : 3644097/5314893,9 = 
#children : 78284/1725389,7 =

vacc_cov1 <- 0.0127                  # vaccine coverage in children
vacc_cov2 <- 0.5969                  # vaccine coverage in adults
vacc_cov3 <- 0.8439                  # vaccine coverage in the elderly

vacc_eff3 <- 0.5                # vaccine efficacy in the elderly (100% in the other age groups)

# Effective vaccine coverage for each age group:
p1 <- vacc_cov1
p2 <- vacc_cov2
p3 <- vacc_cov3 * vacc_eff3

# Population size in total and for each age group:
N <- 1000000
N1 <- 0.2*N
N2 <- 0.65*N
N3 <- 0.15*N

# Fill in initial state values for a naive population based on effective vaccine coverage:

# Initial state values : based on number of infected people
# on the 1st of July (https://www.covid19.admin.ch/fr/epidemiologic/case?rel=abs)
# < 20: 237 infected in Switzerland. 
I1_0107_CH <- 237
# 20-64: 758
I2_0107_CH <- 758
# > 64: 45
I3_0107_CH <- 45
# On N people : N_infected_age_group*N_total/N_pop_suisse
N_suisse <- 8637000
I1_0107 <- I1_0107_CH*N/N_suisse
I2_0107 <- I2_0107_CH*N/N_suisse
I3_0107 <- I3_0107_CH*N/N_suisse


initial_state_values <- c(S1 = N1-p1*N1,
                          S2 = N2-p2*N2,  
                          S3 = N3-p3*N3,
                          I1 = I1_0107,        # the outbreak starts with 1 infected person (can be of either age) 
                          I2 = I2_0107,
                          I3 = I3_0107,
                          R1 = p1*N1,
                          R2 = p2*N2,   
                          R3 = p3*N3)

# Run model output
output <- as.data.frame(ode(y = initial_state_values, 
                            times = times, 
                            func = sir_age_model,
                            parms = parameters))

# Calculate cumulative incidence in each age group:
results1 <- data.frame(child_cum_inc = output$S1[1]-output$S1[nrow(output)],
                       adult_cum_inc = output$S2[1]-output$S2[nrow(output)], 
                       elderly_cum_inc =  output$S3[1]-output$S3[nrow(output)],
                       total_cum_inc = sum(output[1,c("S1", "S2", "S3")])-sum(output[nrow(output),c("S1", "S2", "S3")]))
print(results1)
# works until there ########################################
model_cumulative <- cumsum(output[c("I1")] + output[c("I2")] + output[c("I3")])
plot.default(model_cumulative, type = 'l' ,col="green", xlab = 'time', ylab = 'cumulative number of cases of the model')

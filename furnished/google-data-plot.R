#use Google infection data
library(readxl)
#epidemiology <- read.csv("~/EPFL/labimmersion/data/epidemiology.csv")

#partie 2

#prendre uniquement les cas confirm�s
#prendre uniquement les 3 premiers mois ?
confirmed_google_ <- epidemiology[c("cumulative_confirmed")]
confirmed_google <- confirmed_google_[c(4501:4591),]
#enlever les cas des jours d'avant pour ne prendre que les cas additionnels chaque jour
confirmed_google_norm <- confirmed_google

#vecteur avec les valeurs du mod�le
source("~/Users/cloecortesbalcells/Downloads/Hannah-Mathias-main/modele-calibrage.R")

model_cumulative <- cumsum(output[c("I1")] + output[c("I2")] + output[c("I3")])

#plot les deux courbes l'une sur l'autre
#je fais une fausse deuxi�me courbe pour faire un essai de plot
plot(confirmed_google_norm  + 120.4122,type="l",col="red", xlab = 'days since start of simulation - period: 01.07.2021 to 01.10.2021', ylab = 'cumulative Covid-19 cases number')
lines(model_cumulative + 1525,col="green")
legend(x = 'bottomright', legend = c("SIR model", "Google data"), text.col = c("green4", "red"), trace = TRUE)

res <- chisq.test(model_cumulative + 1525, confirmed_google_norm  + 120.4122)
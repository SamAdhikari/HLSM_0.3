\name{plotDiagnostic}
\alias{plotDiagnostic}
\alias{plotLikelihood}
\alias{plotHLSM.random.fit}
\alias{plotHLSM.fixed.fit}
\alias{plotHLSM.LS}

\title{built-in plot functions for HLSM object}

\description{ 
 plotLikelihood( ) plots the likelihood, and plotDiagnostic( ) plots diagnostic-plot of posterior draws of the parameters from MCMC sample. plotHLSM.random.fit( ) and plotHLSM.fixed.fit( ) are functions to plot mean-results from fitted models, and plotHLSM.LS( ) is for plotting the mean latent position estimates.
}

\usage{
	plotLikelihood(object,burnin = 0, thin = 1)
	plotDiagnostic(chain)
	plotHLSM.random.fit(fitted.model,parameter,burnin=0,thin=1)
	plotHLSM.fixed.fit(fitted.model, parameter,burnin=0,thin=1)
	plotHLSM.LS(fitted.model,pdfname=NULL,burnin=0,thin=1,...)
}


\arguments{

	\item{object}{object of class 'HLSM' obtained as an output from \code{HLSMrandomEF()} or \code{HLSMfixedEF()}
}
	\item{fitted.model}{model fit from either HLSMrandomEF() or HLSMfixedEF()}
	\item{parameter}{parameter to plot; specified as \code{Beta} for slope coefficients, \code{Intercept} for intercept, and \code{Alpha} for intervention effect}
	\item{pdfname}{character to specify the name of the pdf to save the plot if desired. Default is NULL}
        \item{burnin}{numeric value to burn the chain for plotting the results from the 'HLSM'object }
	\item{thin}{a numeric thinning value}
	\item{chain}{a numeric vector of posterior draws of parameter of interest.}
	\item{...}{other options}
}


\value{
 returns plot objects.
}


\author{Sam Adhikari}

\examples{
#using advice seeking network of teachers in 15 schools
#to fit the data

#Random effect model#
priors = NULL
tune = NULL
initialVals = NULL
niter = 10

random.fit = HLSMrandomEF(Y = ps.advice.mat,FullX = ps.edge.vars.mat,
	initialVals = initialVals,priors = priors,
	tune = tune,tuneIn = FALSE,dd = 2,niter = niter,
	intervention = 0)

plotLikelihood(random.fit)

intercept = getIntercept(random.fit)
dim(intercept) ##is an array of dimension niter by 15
plotDiagnostic(intercept[,1])
plotHLSM.LS(random.fit)
plotHLSM.random.fit(random.fit,parameter = 'Beta')
plotHLSM.random.fit(random.fit,parameter = 'Intercept')
##look at the diagnostic plot of intercept for the first school


}

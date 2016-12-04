\name{HLSMrandomEF}
\alias{HLSMrandomEF}
\alias{HLSMfixedEF}
\alias{print.HLSM}
\alias{print.summary.HLSM}
\alias{summary.HLSM}
\alias{getIntercept}
\alias{getAlpha}
\alias{getLS}
\alias{getLikelihood}
\alias{getBeta}

\title{Function to run the MCMC sampler in random effects model (and HLSMfixedEF for fixed effects model)
}

\description{
    Function to run the MCMC sampler to draw from the posterior distribution of intercept, slopes, latent positions, and intervention effect (if applicable). HLSMrandomEF( ) fits random effects model; HLSMfixedEF( ) fits fixed effects model. 
}

\usage{

HLSMrandomEF(Y,edgeCov=NULL, receiverCov = NULL, senderCov = NULL, 
        FullX = NULL,initialVals = NULL, priors = NULL, tune = NULL,
        tuneIn = TRUE,TT = NULL,dd, niter,intervention)

HLSMfixedEF(Y,edgeCov=NULL, receiverCov = NULL, senderCov = NULL,
        FullX = NULL, initialVals = NULL, priors = NULL, tune = NULL,
        tuneIn = TRUE, TT = NULL,dd, niter,intervention)

getBeta(object, burnin = 0, thin = 1)
getIntercept(object, burnin = 0, thin = 1)
getAlpha(object, burnin = 0, thin = 1)
getLS(object, burnin  = 0, thin = 1)
getLikelihood(object, burnin = 0, thin = 1)
}


\arguments{

\item{Y}{
	input outcome for different networks. Y can either be

        (i). list of socio-matrix for \code{K} different networks

        (ii). list of data frame with columns \code{Sender}, \code{Receiver} and \code{Outcome} for \code{K} different networks

        (iii). a dataframe with columns named as follows: \code{id} to identify network, \code{Receiver} for receiver nodes, \code{Sender} for sender nodes and finally, \code{Outcome} for the edge outcome.       
}

\item{edgeCov}{
	data frame to specify edge level covariates with

        (i). a column for network id named \code{id}, 

        (ii). a column for sender node named \code{Sender},

        (iii). a column for receiver nodes named \code{Receiver}, and 

        (iv). columns for values of each edge level covariates.
}

\item{receiverCov}{
	a data frame to specify nodal covariates as edge receivers with

        (i.) a column for network id named \code{id}, 

        (ii.) a column \code{Node} for node names, and 

        (iii). the rest for respective node level covariates.
}

\item{senderCov}{
	a data frame to specify nodal covariates as edge senders with 

        (i). a column for network id named \code{id}, 

        (ii). a column \code{Node} for node names, and 

        (iii). the rest for respective node level covariates.
}

\item{FullX}{
	list of numeric arrays of dimension \code{n} by \code{n} by \code{p} of covariates for K different networks. When FullX is provided to the function, edgeCov, receiverCov and senderCov must be specified as NULL. 
}
		
\item{initialVals}{
	an optional list of values to initialize the chain. If \code{NULL} default initialization is used, else 
	\code{initialVals = list(ZZ, beta, intercept, alpha)}.
 
	For fixed effect model \code{beta} is a vector of length \code{p} and \code{intercept} is a vector of length 1.

	For random effect model \code{beta} is an array of dimension  \code{K} by \code{p}, and \code{intercept} is a vector of length \code{K}, where \code{p} is the number of covariates and \code{K} is the number of network.

	\code{ZZ} is an array of dimension \code{NN} by \code{dd}, where \code{NN} is the sum of nodes in all \code{K} networks.

	\code{alpha} is a numeric variable and is 0 for no-intervention model.	
}


\item{priors}{
      an optional list to specify the hyper-parameters for the prior distribution of the paramters.
        If priors = \code{NULL}, default value is used. Else,

    \code{priors=}

	\code{list(MuBeta,VarBeta,MuAlpha,VarAlpha,MuZ,VarZ,PriorA,PriorB)}
       
	\code{MuBeta} is a numeric vector of length PP + 1 specifying the mean of prior distribution for coefficients and intercept

	\code{VarBeta} is a numeric vector for the variance of the prior distribution of coefficients and intercept. Its length is same as that of MuBeta.

     \code{MuAlpha} is a numeric variable specifying the mean of prior distribution of intervention effect. Default is 0.

    \code{VarAlpha} is a numeric variable for the variance of the prior distribution of intervention effect. Default is 100.

    \code{MuZ} is a numeric vector of length same as the dimension of the latent space, specifying the prior mean of the latent positions.
 
    \code{VarZ} is a numeric vector of length same as the dimension of the latent space, specifying diagonal of the variance covariance matrix of the prior of latent positions.

    \code{PriorA, PriorB} is a numeric variable to indicate the rate and scale parameters for the inverse gamma prior distribution of the hyper parameter of variance of slope and intercept
       }


\item{tune}{
    an optional list of tuning parameters for tuning the chain. If tune = \code{NULL}, default tuning is done. Else, 

	\code{tune = list(tuneAlpha, tuneBeta, tuneInt,tuneZ)}.

	\code{tuneAlpha}, \code{tuneBeta} and \code{tuneInt} have the same structure as \code{beta}, \code{alpha} and \code{intercept} in \code{initialVals}.

	 \code{ZZ} is a vector of length \code{NN}.
}
    \item{tuneIn}{
    a logical to indicate whether tuning is needed in the MCMC sampling. Default is \code{FALSE}.
}

\item{TT}{
     a vector of binaries to indicate treatment and control networks. If there is no intervention effect, TT = \code{NULL} (default).
}


\item{dd}{
    dimension of latent space.
}

\item{niter}{
    number of iterations for the MCMC chain.
}

\item{intervention}{
    binary variable indicating whether the posterior distribution of the intervention effect is to be estimated.
}

\item{object}{
	object of class 'HLSM' returned by \code{HLSM()} or \code{HLSMfixedEF()}
}

\item{burnin}{
	numeric value to burn the chain while extracting results from the 'HLSM'object. Default is \code{burnin = 0}. 
}

\item{thin}{
	numeric value by which the chain is to be thinned while extracting results from the 'HLSM' object. Default is \code{thin = 1}. 
} 
}




\value{
    Returns an object of class "HLSM". It is a list with following components:
    \item{draws}{
       list of posterior draws for each parameters.
    }
    \item{acc}{
      list of acceptance rates of the parameters.
    }
    \item{call}{
    the matched call.
 }

\item{tune}{
	final tuning values
}	

}

\author{
    Sam Adhikari
}

\references{Tracy M. Sweet, Andrew C. Thomas and Brian W. Junker (2012), "Hierarchical Network Models for Education Research: Hierarchical Latent Space Models", Journal of Educational and Behavorial Statistics.
}


\examples{

library(HLSM)

#Set values for the inputs of the function
priors = NULL
tune = NULL
initialVals = NULL
niter = 10

#Random effect HLSM on Pitt and Spillane data
random.fit = HLSMrandomEF(Y = ps.advice.mat,FullX = ps.edge.vars.mat,
	initialVals = initialVals,priors = priors,
	tune = tune,tuneIn = FALSE,dd = 2,niter = niter,
	intervention = 0)


summary(random.fit)
names(random.fit)

#extract results without burning and thinning
Beta = getBeta(random.fit)
Intercept = getIntercept(random.fit)
LS = getLS(random.fit)
Likelihood = getLikelihood(random.fit)

##Same can be done for fixed effect model

#Fixed effect HLSM on Pitt and Spillane data 

fixed.fit = HLSMfixedEF(Y = ps.advice.mat,FullX = ps.edge.vars.mat,
	initialVals = initialVals,priors = priors,
	tune = tune,tuneIn = FALSE,dd = 2,niter = niter,
	intervention = 0)

summary(fixed.fit)
names(fixed.fit)

}




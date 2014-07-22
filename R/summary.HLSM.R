summary.HLSM = function(object,...){
	if(class(object) != 'HLSM'){stop('input must be of class HLSM')}

	call = object$call

	Betas = getBeta(object, ...)
	if(length(dim(Betas)) == 3){
		est.slopes = lapply(1:dim(Betas)[[3]],
			function(x){ t(sapply(1:ncol(Betas[,,x]),
			function(y) data.frame(min = round(min(Betas[,y,x]),3),
				max = round(max(Betas[,y,x]),3),
				est.mean = round(mean(Betas[,y,x]),3), 
				sd = round(sd(Betas[,y,x]),3), 
				q.25 = round(quantile(Betas[,y,x],0.025),3),
				q.975 = round(quantile(Betas[,y,x],0.975),3))))})
	}else{
		Betas = as.matrix(Betas)
		est.slopes = t(sapply(1:ncol(Betas),function(y) data.frame(min = round(min(Betas[,y]),3), max = round(max(Betas[,y]),3),est.mean = round(mean(Betas[,y]),3), sd = round(sd(Betas[,y]),3), q.25 = round(quantile(Betas[,y],0.025),3), q.975 = round(quantile(Betas[,y],0.975),3)) ) )
	}

	Intercept = getIntercept(object,...)
	if(class(Intercept) == 'matrix'){
		est.intercept = t(sapply(1:ncol(Intercept), function(y) data.frame(min = round(min(Intercept[,y]),3), max = round(max(Intercept[,y]),3), est.mean = round(mean(Intercept[,y]),3), sd = round(sd(Intercept[,y]),3), q.25 = round(quantile(Intercept[,y],0.025),3), q.975 = round(quantile(Intercept[,y],0.975),3) )) )
	}else{
		est.intercept = data.frame(min = round(min(Intercept),3), max = round(max(Intercept),3), est.mean = round(mean(Intercept),3), sd = round(sd(Intercept),3), q.025 = round(quantile(Intercept,0.025),3), q.975 = round(quantile(Intercept, 0.975),3) )
      }
	
	Alphas = getAlpha(object,...)
	if(any(Alphas != 0)){
			est.alpha = data.frame(min = round(min(Alphas),3), max = round(max(Alphas),3), est.mean = round(mean(Alphas),3), sd = round(sd(Alphas),3), q.025 = round(quantile(Alphas,0.025),3), q.975 = round(quantile(Alphas, 0.975),3) )
		res =  list(call = object$call,
         est.intercept = est.intercept, est.alpha = est.alpha, est.slopes = est.slopes)
	}else{
		res = list(call = object$call,
         est.intercept = est.intercept, est.slopes = est.slopes)}

    class(res) = 'summary.HLSM'
    res
}


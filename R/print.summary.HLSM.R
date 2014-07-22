print.summary.HLSM = function(x,...){
    message("Call:\n")
    print(x$call)
    message("\n Estimated Intercept:\n")
    print(x$est.intercept)
    message("\n Estimated Slopes:\n")
    print(x$est.slopes)
}


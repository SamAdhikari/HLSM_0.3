#include <R.h>
#include <R_ext/Print.h>
#include <iostream>
#include <math.h>
#include <Rmath.h>

//small functions used later in MCMC
double square(double x){
    double y = x*x;
    return y ;
}
double logitInverse(double x){
	return 1.0/(1.0 + exp(-1.0 * x));
}

//read row ii of a matrix with dd columns
void ReadRow(double *XX, int ii, double *Row, int *nn, int *dd)
{
	for(int kk =0; kk < dd[0]; kk++)
	{
		Row[kk] = XX[ii + kk*nn[0]];
	}
}
//write row ii in a matrix
void WriteRow(double *XX, int ii, double *Xvec, int *nn, int *dd)
{
	for(int kk = 0; kk < dd[0]; kk ++)
	{
		XX[ii + kk*nn[0]] = Xvec[kk];
	}
}
//extract Y and Z matrix for KK network with nn rows and pp columns


void getY(double *Mat,double *NewMat, int *nn,int kk){
	int ss = 0.0;
	if(kk > 0){
	for(int ll = 0;ll < kk; ll++){
		ss += nn[ll]*nn[ll];
	}
	}
	for(int xx= 0;xx < nn[kk]; xx++){
		for(int yy =0; yy < nn[kk]; yy++){
			NewMat[xx+yy*nn[kk]] = Mat[xx+yy*nn[kk]+ss];
			}
		}
}

void getZ(double *Mat,double *NewMat, int *nn,int dd,int kk){
	int ss = 0.0;
	if(kk > 0){
	for(int ll = 0;ll < kk; ll++){
		ss += (dd*nn[ll]);
		}
	}
	for(int xx= 0;xx < nn[kk]; xx++){
		for(int yy =0; yy < dd; yy++){
			NewMat[xx+yy*nn[kk]] = Mat[xx+yy*nn[kk]+ss];
			}
		}
}
	
//ss = sum(nn[1:kk] for kk groups
void readX(double *X,double *newX, int *nn, int pp, int kk){
    	int ss = 0.0;
	if(kk > 0){
	for(int ll = 0;ll < kk; ll++){
		ss += nn[ll]*nn[ll];
	}
	}
	for(int xx = 0; xx < nn[kk]; xx++){
		for(int yy = 0; yy < nn[kk];yy ++){
			for(int zz =0; zz <pp; zz++){
				newX[xx+yy*nn[kk]+zz*nn[kk]*nn[kk]] = X[xx+yy*nn[kk]+zz*nn[kk]*nn[kk]+pp*ss];
			}
		}
	}
}
//compute distance matrix
void distMat(int *nn, int *dd, double *ZZ, double *dMat){
    int ii,jj,kk;
    double tmp;
    for(ii = 0 ; ii <= (nn[0]-1) ; ii++){
      for(jj = 0 ; jj <= ii ; jj++){
	      tmp = 0.0;
	for(kk = 0 ; kk < dd[0] ; kk++){
		double VV = ZZ[ii+kk*nn[0]] - ZZ[jj+kk*nn[0]];
		tmp = tmp + square(VV);
	}
	dMat[jj*nn[0]+ii] = sqrt(tmp);
	dMat[ii*nn[0]+jj] = sqrt(tmp);
      }
    }
}

//Compute logpriors 
void LogpriorBeta(double *Beta, double *MuBeta, double *SigmaBeta, double *val)
{
    double sigma = sqrt(SigmaBeta[0]);	
    val[0] = dnorm(Beta[0], MuBeta[0], sigma, 1);
}

void LogpriorAlpha(double *Alpha, double *MuAlpha, double *SigmaAlpha, double *val)	 
{
	double sigma = sqrt(SigmaAlpha[0]);
	val[0] = dnorm(Alpha[0],MuAlpha[0],sigma,1);
}

//Z is treated as independent normals in each coordinate
void LogpriorZZ(double *ZZ, double *Mu, double *Var, int *dd, double *val){
    int ii;
    double total = 0.0;
    for(ii = 0 ; ii < dd[0] ; ii++){
      double sigma = sqrt(Var[ii]);	    
      total = total + dnorm(ZZ[ii],Mu[ii],sigma,1);
    }
    val[0] = total;
}

//compute loglikelihood for a single network
void FullLogLik(double *beta, double *YY, double *XX, double *ZZ, double *alpha, int *Tr, double *intercept,int *nn, int *pp, int *dd, double *Val){
	double* dMat = 0;
	dMat = new double[nn[0]*nn[0]];
	distMat(nn,dd,ZZ,dMat);
	double total = 0.0;
	double tmp1,tmp2, tmpVal1, tmpVal2;
	for(int ii = 1; ii < nn[0]; ii++){
		for(int jj = 0; jj < ii; jj++){
			tmpVal1 = 0.0;
			tmpVal2 = 0.0;
			//compute t(X)*beta for all P
			for(int kk = 0; kk < pp[0]; kk++){
				tmpVal1 = tmpVal1 + beta[kk]*XX[ii+jj*nn[0]+kk*nn[0]*nn[0]];
				tmpVal2 = tmpVal2 + beta[kk]*XX[jj + ii*nn[0] + kk*nn[0]*nn[0]];
			}
			double vv1 = (intercept[0] + tmpVal1 + alpha[0]*Tr[0] - dMat[jj*nn[0] + ii]); 
			double vv2 = (intercept[0] + tmpVal2 + alpha[0]*Tr[0] - dMat[jj+ii*nn[0]]);
			tmp1 = logitInverse(vv1);
			tmp2 = logitInverse(vv2);
			if(YY[jj*nn[0] + ii] == 1){
				total = total + log(tmp1);
			}else if(YY[jj*nn[0] + ii] == 0){
				total = total + log(1.0 - tmp1);
			}
			if(YY[jj + ii*nn[0]] == 1){
				total = total + log(tmp2);
			}else if(YY[jj + ii*nn[0]] == 0){
				total = total + log(1.0 - tmp2);
			}
		}		
	}
	Val[0] = total;
	delete[] dMat;
}

//sum of loglikelihoods for all KK networks
void AllLogLik(double *X, double *Y, double *Z, int *T, int *nn, int *pp, int *dd, int *KK, double *beta, double *intercept,double *alpha, double *lliknew)
{
	double Val;
	int slen = KK[0]+1;
	double* sumn = 0; // need this to store the updated parameters
	sumn = new double[slen];
	sumn[0] = 0; 
	for(int iz =0; iz < KK[0]; iz++){
		sumn[iz+1] = sumn[iz] + nn[iz];
	}
	lliknew[0] = 0.0;
	for(int kk = 0; kk < KK[0]; kk++){
         	 int ss = sumn[kk];
		 double* XMat = 0;
	   	 XMat = new double[nn[kk]*nn[kk]*pp[0]];
		 double* YMat = 0;
		 YMat = new double[nn[kk]*nn[kk]];
		 double* ZMat = 0;
		 ZMat = new double[nn[kk]*dd[0]];
		 getY(Y,YMat,nn,kk);
		 readX(X,XMat,nn,pp[0],kk);
		 getZ(Z,ZMat,nn,dd[0],kk);
		 //use the X Y and Z for that group.
	         FullLogLik(beta, YMat, XMat, ZMat, alpha, &T[kk], intercept,&nn[kk],pp,dd,&Val);
		 lliknew[0] = lliknew[0] + Val;
		 delete[] XMat;
	 	 delete[] YMat;
	 	 delete[] ZMat;
	 }
	delete[] sumn;
}

		

//update Z as an independent normal
void updateZ(double *XX,double *YY,double *ZZ,int *TT,int *nn, int *pp,int *dd,double *beta,double *intercept,double *alpha,double *zzPrior,double *Var,double *tuneZ,double *llOld,double *accZ)
{  
     double draw, logRR;
     double* Znew = 0;
     Znew = new double[nn[0]*dd[0]];
     double lpNew;
     double llNew;
     double lpOld;
     for(int ss = 0 ; ss < nn[0]*dd[0] ; ss++){
         Znew[ss] = ZZ[ss];
    } 
     for(int ii = 1 ; ii < nn[0] ; ii++){ //keep the first entry of Z fixed
	      double* ZZsm = 0;
	      ZZsm = new double[dd[0]];
//	      double ZZsm[dd[0]];
	      ReadRow(ZZ,(ii),ZZsm,nn,dd);
	      LogpriorZZ(ZZsm,zzPrior,Var,dd,&lpOld);
	      double* ZnewSm = 0;
	      ZnewSm = new double[dd[0]];
//      	      double ZnewSm[dd[0]];
      	      for(int kk = 0 ; kk < dd[0] ; kk++){ 
			  ZnewSm[kk]  = ZZsm[kk] + tuneZ[ii]*rnorm(0.0,1.0);
			  if(ii == 1  && kk == 0){ //Positivity constraint on Znew[2,1]
				  if(ZnewSm[kk] < Znew[0]){
					  ZnewSm[kk] = -1.0*(ZnewSm[kk]-Znew[0])+1.0;
				  }
			  }
			  if(ii == 1 && kk == 1){ //set Znew[2,2] = 0
			      ZnewSm[kk] = 0.0;
		          }
                          if((kk == 1 && ii == 2)){ //Positivity constraint on Znew[3,2]
		      		  if(ZnewSm[kk] < 0){
		  			  ZnewSm[kk] = -1.0 * ZnewSm[kk];
		      		  }
			  }		
	      	      }
	      WriteRow(Znew, (ii), ZnewSm,nn,dd);
    	      FullLogLik(beta, YY, XX, Znew, alpha, TT, intercept,nn,pp,dd,&llNew);
    	      LogpriorZZ(ZnewSm,zzPrior,Var,dd,&lpNew);
    	      logRR = llNew - llOld[0] + lpNew - lpOld;
    	      draw = runif(0.0,1.0);
    	      if(log(draw) < logRR){
		      WriteRow(ZZ, (ii), ZnewSm,nn,dd);
	  	      llOld[0] = llNew;	  	   
		      accZ[ii] = accZ[ii] + 1;
	      }else{
	  	      WriteRow(Znew,ii,ZZsm,nn,dd);
	  }
	      delete[] ZnewSm;
	      delete[] ZZsm;
     }
     delete[] Znew;
}    



//UPDATES FOR RANDOM EFFECT MODEL
//update beta 
void updateBetamulti(double *X,double *Y,double *Z,int *T,int *nn,int *pp,int *dd, double *beta, double *intercept,double *alpha, double *mu,double *sigmasq,double *tune,double *llik,double *acc)
{
	double* betanew = 0;
	betanew = new double[pp[0]];
//	double betanew[pp[0]];
	for(int k =0;k<pp[0];k++){
		betanew[k] = beta[k];
	}
	double lpBetanew;
	double llikBetanew;
	double lpBeta;
	for(int ii = 0; ii < pp[0]; ii++){
		LogpriorBeta(&beta[ii], &mu[ii], &sigmasq[ii],&lpBeta); 
		betanew[ii] = beta[ii] + tune[ii]*rnorm(0.0,1.0);
		LogpriorBeta(&betanew[ii],&mu[ii],&sigmasq[ii],&lpBetanew);
		FullLogLik(betanew, Y, X, Z, alpha, T, intercept,nn, pp, dd, &llikBetanew);
		double logratio = lpBetanew - lpBeta + llikBetanew - llik[0];
		if(log(runif(0.0,1.0)) < logratio){
			beta[ii] = betanew[ii];
			llik[0] = llikBetanew;
		//	lpBeta[ii] = lpBetanew;
			acc[ii] = acc[ii] + 1;
		}else{                    //if not accepted reset betanew[ii] with old beta
		betanew[ii] = beta[ii];
	       }
	}
        delete[] betanew;
}

//update intercept
void updateIntercept(double *X,double *Y,double *Z,int *T,int *nn,int *pp,int *dd,double *beta,double *intercept,double *alpha,double *mu,double *sigmasq,double *tuneInt,double *lpInt,double *llik,double *acc){
	double intnew;
	double lpIntNew;
	double llikIntnew;
       	intnew = intercept[0] + tuneInt[0]*rnorm(0.0,1.0);
	LogpriorBeta(&intnew, mu,sigmasq,&lpIntNew);
	FullLogLik(beta,Y,X,Z,alpha,T,&intnew,nn,pp,dd,&llikIntnew);
	double logratio = lpIntNew-lpInt[0]+llikIntnew-llik[0];
	if(log(runif(0.0,1.0)) < logratio){
		intercept[0] = intnew;
		llik[0] = llikIntnew;
		lpInt[0] = lpIntNew;
		acc[0] = acc[0] + 1;
	}
}


//UPDATES FOR FIXED EFFECT MODEL
//update beta 
void updateBetamultiFixedEF(double *X,double *Y,double *Z,int *T,int *nn,int *pp,int *dd, int *KK, double *beta, double *intercept,double *alpha, double *mu,double *sigmasq,double *tune,double *llik,double *acc)
{
	double* betanew = 0;
	betanew = new double[pp[0]];
//	double betanew[pp[0]];
	for(int k =0;k<pp[0];k++){
		betanew[k] = beta[k];
	}
	double lpBetanew;
	double llikBetanew;
	double lpBeta;
	for(int ii = 0; ii < pp[0]; ii++){
		LogpriorBeta(&beta[ii], &mu[ii], &sigmasq[ii],&lpBeta); 
		betanew[ii] = beta[ii] + tune[ii]*rnorm(0.0,1.0);
		LogpriorBeta(&betanew[ii],&mu[ii],&sigmasq[ii],&lpBetanew);
        	AllLogLik(X, Y, Z, T, nn, pp, dd,KK, betanew, intercept,alpha, &llikBetanew);
		double logratio = lpBetanew - lpBeta + llikBetanew - llik[0];
		if(log(runif(0.0,1.0)) < logratio){
			beta[ii] = betanew[ii];
			llik[0] = llikBetanew;
		//	lpBeta[ii] = lpBetanew;
			acc[ii] = acc[ii] + 1;
		}else{                    //if not accepted reset betanew[ii] with old beta
		betanew[ii] = beta[ii];
	       }
	}
	delete[] betanew;
}

//update intercept
void updateInterceptFixedEF(double *X,double *Y,double *Z,int *T,int *nn,int *pp,int *dd,int *KK, double *beta,double *intercept,double *alpha,double *mu,double *sigmasq,double *tuneInt,double *lpInt,double *llik,double *acc){
	double intnew;
	double lpIntnew;
	double llikIntnew;
       	intnew = intercept[0] + tuneInt[0]*rnorm(0.0,1.0);
	LogpriorBeta(&intnew, mu,sigmasq,&lpIntnew);
	AllLogLik(X, Y, Z, T, nn, pp, dd,KK, beta, &intnew,alpha,&llikIntnew);
	double logratio = lpIntnew-lpInt[0]+llikIntnew-llik[0];
	if(log(runif(0.0,1.0)) < logratio){
		intercept[0] = intnew;
		llik[0] = llikIntnew;
		lpInt[0] = lpIntnew;
		acc[0] = acc[0] + 1;
	}
}



/////
//update alpha
void updateAlpha(double *X,double *Y,double *Z,int *T,int *nn,int *pp,int *dd,int *KK,double *beta, double *intercept,double *alpha,double *mu,double *sigmasq,double *tuneAlpha,double *lpAlpha,double *llikAll,double *accalpha)
{
	 double Alphanew = alpha[0] + tuneAlpha[0]*rnorm(0.0,1.0);
	 double llikAlphanew;
	 double lpAlphanew;
	 AllLogLik(X, Y, Z, T, nn, pp, dd,KK, beta, intercept,&Alphanew,&llikAlphanew);
	 LogpriorAlpha(&Alphanew, mu, sigmasq,&lpAlphanew);
	 double logratio = llikAlphanew - llikAll[0] + lpAlphanew -lpAlpha[0];
	 if(log(runif(0.0,1.0)) < logratio){
		 alpha[0] = Alphanew;
		 lpAlpha[0] = lpAlphanew;
		 accalpha[0] = accalpha[0] + 1;
	 }
}

/////////////////////////////////

//SAMPLER FOR FIXED EFFECT MODEL
extern "C" {

void sampleFixedIntervention(int *niter, double *XX,double *YY,double *ZZ,int *TT,int *nn,int *PP,int *dd,int *KK,double *beta,double *intercept,double *alpha,double *MuAlpha,double *SigmaAlpha,double *MuBeta, double *SigmaBeta, double *MuZ,double *VarZ, double *tuneBetaAll,double *tuneInt,double *tuneAlpha,double *tuneZAll,double *accBetaAll,double *accAlpha, double *accIntAll,double *accZAll, double *betaFinal, double *AlphaFinal, double *ZZFinal, double *InterceptFinal,double *Zvar1,double *Zvar2,double *likelihood,double *PriorA,double *PriorB, int *intervention){
	int slen = KK[0]+1;
	double* sumn = 0; // need this to store the updated parameters
	sumn = new double[slen];
	sumn[0] = 0; 
	for(int iz =0; iz < KK[0]; iz++){
		sumn[iz+1] = sumn[iz] + nn[iz];
	}
	int ll = KK[0];
	int sumAll = sumn[ll];
//	int lenY = sumn[KK[0]+1];
	double lpAlpha,lpInt,llik;
//	double lpZ, lpBeta;
        double* D = 0;
	D = new double[dd[0]];	
//	double D[dd[0]];
	double C;
	double muInt = MuBeta[PP[0]];
      	double sigmaInt = SigmaBeta[PP[0]];

	for(int ii = 0; ii < niter[0]; ii++){
		GetRNGstate();
		LogpriorAlpha(alpha, MuAlpha, SigmaAlpha,&lpAlpha);
		//Recursion for KK groups
		double llikall = 0.0;
        	for(int ss = 0.0;ss < dd[0];ss++){
			D[ss] = 0.0;
		}
		for(int kk = 0; kk < KK[0]; kk++){
			int ss = sumn[kk];
			int ss2 = sumn[kk + 1];
		        double* accZ = 0;
                       	accZ = new double[nn[kk]];	
			double* tuneZ = 0;
                       	tuneZ = new double[nn[kk]];
			double* XMat = 0;
			XMat = new double[nn[kk]*nn[kk]*PP[0]];
			double* YMat = 0;
			YMat = new double[nn[kk]*nn[kk]];
			double* ZMat = 0;
			ZMat = new double[nn[kk]*dd[0]];
			//read X, Y, X for a network
                        getY(YY,YMat,nn,kk);
			readX(XX,XMat,nn,PP[0],kk);
			getZ(ZZ,ZMat,nn,dd[0],kk);
		
			for(int zz =0;zz < nn[kk];zz++){
				accZ[zz] = accZAll[zz+ss];
				tuneZ[zz] = tuneZAll[zz+ss];
			}
			
       			//loglikelihood for the group
	               	FullLogLik(beta, YMat, XMat, ZMat,alpha, &TT[kk],intercept, &nn[kk],PP,dd,&llik);

			updateZ(XMat,YMat,ZMat,&TT[kk],&nn[kk],PP,dd,beta,intercept,alpha,MuZ,VarZ,tuneZ,&llik,accZ);
		       //store
       	          	for(int nZ=0; nZ < nn[kk]; nZ++){
       				for(int dZ=0; dZ < dd[0]; dZ++){
					ZZ[nZ+dZ*nn[kk]+dd[0]*ss] = ZMat[nZ+dZ*nn[kk]];
				       // ZZFinal[nZ+ss+ss2*dZ+ii*sumAll*dd[0]] = ZMat[nZ+dZ*nn[kk]];
					ZZFinal[nZ+dZ*nn[kk]+dd[0]*ss+ii*sumAll*dd[0]] = ZMat[nZ+dZ*nn[kk]];
					D[dZ] += square(ZMat[nZ+dZ*nn[kk]] - 0.0);
				}
					accZAll[nZ+ss] = accZ[nZ];
       				}       			
			llikall += llik;
			delete[] XMat;
			delete[] YMat;
			delete[] ZMat;
			delete[] tuneZ;
			delete[] accZ;
		}
		
//	Rprintf("%f",llikall);

		LogpriorBeta(intercept, &muInt, &sigmaInt,&lpInt); //logprior of intercept
                updateInterceptFixedEF(XX,YY,ZZ,TT,nn,PP,dd,KK,beta,intercept,alpha,&muInt,&sigmaInt,tuneInt,&lpInt,&llikall, accIntAll);
		updateBetamultiFixedEF(XX,YY,ZZ,TT,nn,PP,dd,KK, beta,intercept,alpha,MuBeta,SigmaBeta,tuneBetaAll,&llikall, accBetaAll);
								        //store the updated values
		InterceptFinal[ii] = intercept[0];    
       		for(int pp = 0; pp < PP[0]; pp++){
       			betaFinal[ii+pp*niter[0]] = beta[pp];
       			}
			

	//update alpha
      	    if(intervention[0] == 1){
	        	updateAlpha(XX,YY,ZZ,TT,nn,PP,dd,KK,beta,intercept,alpha,MuAlpha,SigmaAlpha,tuneAlpha,&lpAlpha,&llikall,accAlpha);
		        AlphaFinal[ii] = alpha[0];
	}

    //Update varaince for ZZ    
	
	for(int vv = 0; vv < dd[0]; vv++){	
		D[vv] = D[vv]/2.0 + PriorB[0];
		C = PriorA[0] + (sumAll)/2.0;
		VarZ[vv] = 1.0/rgamma(C,1.0/D[vv]);
	}
	
	Zvar1[ii] = VarZ[0];
	Zvar2[ii] = VarZ[1];
	likelihood[ii] = llikall;
	PutRNGstate();
	}
		 
	delete[] sumn;
	delete[] D;
}



/////////////////////////////////////////////////////
///////////////SAMLPE FOR RANDOM EFFECT MODEL////////
//////////////////////////////////////////////////////
void sampleRandomIntervention(int *niter, double *XX,double *YY,double *ZZ,int *TT,int *nn,int *PP,int *dd,int *KK,double *beta,double *intercept,double *alpha,double *MuAlpha,double *SigmaAlpha,double *MuBeta, double *SigmaBeta, double *MuZ,double *VarZ, double *tuneBetaAll,double *tuneInt,double *tuneAlpha,double *tuneZAll,double *accBetaAll,double *accAlpha, double *accIntAll,double *accZAll, double *betaFinal, double *AlphaFinal, double *ZZFinal, double *InterceptFinal,double *Zvar1,double *Zvar2,double *postVar, double *postMu, double *likelihood,double *PriorA,double *PriorB, int *intervention){
	//Set starting points
//	InitialVal(XX,YY,ZZ,nn,TT,dd,PP,KK,beta,intercept,alpha,intervention);
	int slen = KK[0]+1;
	double* sumn = 0; // need this to store the updated parameters
	sumn = new double[slen];
	sumn[0] = 0; 
	for(int iz =0; iz < KK[0]; iz++){
		sumn[iz+1] = sumn[iz] + nn[iz];
	}
	int ll = KK[0];
	int sumAll = sumn[ll];
//	int lenY = sumn[KK[0]+1];
	double lpAlpha,lpInt,llik; 
	//double lpZ, lpBeta;
        double* accbeta = 0;
        accbeta = new double[PP[0]];
        double accInt = 0;
	double* tuneBeta = 0;
	tuneBeta = new double[PP[0]];
	double* D = 0;
	D = new double[dd[0]];
//	double D[dd[0]];
	double C;
	for(int ii = 0; ii < niter[0]; ii++){
		double muInt = MuBeta[PP[0]];
        	double sigmaInt = SigmaBeta[PP[0]];
		GetRNGstate();
		LogpriorAlpha(alpha, MuAlpha, SigmaAlpha,&lpAlpha);
		//Recursion for KK groups
		double llikall = 0.0;
        	for(int ss = 0.0;ss < dd[0];ss++){
			D[ss] = 0.0;
		}
		for(int kk = 0; kk < KK[0]; kk++){
			int ss = sumn[kk];
			int ss2 = sumn[kk + 1];
		        double* accZ = 0;
                       	accZ = new double[nn[kk]];	
			double* tuneZ = 0;
                       	tuneZ = new double[nn[kk]];
			double* XMat = 0;
			XMat = new double[nn[kk]*nn[kk]*PP[0]];
			double* YMat = 0;
			YMat = new double[nn[kk]*nn[kk]];
			double* ZMat = 0;
			ZMat = new double[nn[kk]*dd[0]];
			double* betaKK = 0;
			betaKK = new double[PP[0]];
			//read X, Y, X for a network
                        getY(YY,YMat,nn,kk);
			readX(XX,XMat,nn,PP[0],kk);
			getZ(ZZ,ZMat,nn,dd[0],kk);

           		//make sure beta has pp rows and KK columns
			for(int ll = 0; ll<PP[0];ll++){
				betaKK[ll] = beta[ll+kk*PP[0]];
				accbeta[ll] = accBetaAll[ll+kk*PP[0]];
				tuneBeta[ll] = tuneBetaAll[ll+kk*PP[0]];
				//betaKK[ll] = beta[kk+ll*KK[0]];
				//accbeta[ll] = accBetaAll[kk+ll*KK[0]];
				//tuneBeta[ll] = tuneBetaAll[kk+ll*KK[0]];

			}
			accInt = accIntAll[kk];
			
			for(int zz =0;zz < nn[kk];zz++){
				accZ[zz] = accZAll[zz+ss];
				tuneZ[zz] = tuneZAll[zz+ss];
			}
			
       			//loglikelihood for the group
	               	FullLogLik(betaKK, YMat, XMat, ZMat,alpha, &TT[kk], &intercept[kk],&nn[kk], PP, dd,&llik);
			LogpriorBeta(&intercept[kk], &muInt, &sigmaInt,&lpInt); //logprior of intercept
			updateBetamulti(XMat,YMat,ZMat,&TT[kk],&nn[kk],PP,dd,betaKK,&intercept[kk],alpha,MuBeta,SigmaBeta,tuneBeta,&llik, accbeta);
			updateZ(XMat,YMat,ZMat,&TT[kk],&nn[kk],PP,dd,betaKK,&intercept[kk],alpha,MuZ, VarZ,tuneZ,&llik,accZ);	
			updateIntercept(XMat,YMat,ZMat,&TT[kk],&nn[kk],PP,dd,betaKK,&intercept[kk],alpha,&muInt,&sigmaInt,&tuneInt[kk],&lpInt,&llik, &accInt);
				
		        //store the updated values
			InterceptFinal[ii+kk*niter[0]] = intercept[kk];    
       			for(int pp = 0; pp < PP[0]; pp++){
				beta[pp+kk*PP[0]] = betaKK[pp];
       				betaFinal[ii+pp*niter[0]+kk*PP[0]*niter[0]] = betaKK[pp];
				accBetaAll[pp+kk*PP[0]] = accbeta[pp];
   						}
		       
       	          	for(int nZ=0; nZ < nn[kk]; nZ++){
       				for(int dZ=0; dZ < dd[0]; dZ++){
					ZZ[nZ+dZ*nn[kk]+dd[0]*ss] = ZMat[nZ+dZ*nn[kk]];
				//        ZZFinal[nZ+ss+ss2*dZ+ii*sumAll*dd[0]] = ZMat[nZ+dZ*nn[kk]];
					ZZFinal[nZ+dZ*nn[kk]+dd[0]*ss+ii*sumAll*dd[0]] = ZMat[nZ+dZ*nn[kk]];
					D[dZ] += square(ZMat[nZ+dZ*nn[kk]] - 0.0);
				}
					accZAll[nZ+ss] = accZ[nZ];
       				}       			
			accIntAll[kk] = accInt;
			llikall += llik;
			delete[] XMat;
			delete[] YMat;
			delete[] ZMat;
			delete[] betaKK;
			delete[] tuneZ;
			delete[] accZ;
		}
	//update alpha
//	Rprintf("%f",llikall);

	if(intervention[0] == 1){
		updateAlpha(XX,YY,ZZ,TT,nn,PP,dd,KK,beta,intercept,alpha,MuAlpha,SigmaAlpha,tuneAlpha,&lpAlpha,&llikall,accAlpha);
		AlphaFinal[ii] = alpha[0];
	}

	//update mu and sigmasq for beta
	for(int ee = 0;ee< PP[0];ee++){
		double denom = SigmaBeta[ee] + KK[0];
		double sumbeta = 0.0;
		for(int ww = 0;ww < KK[0];ww++){
			sumbeta += beta[ee+ww*PP[0]];
		}
		double tempmu = (sumbeta + SigmaBeta[ee]*MuBeta[ee])/denom;
		double tempssq = SigmaBeta[ee]/denom;
		MuBeta[ee] = rnorm(tempmu,tempssq);
		postMu[ii+niter[0]*ee] = MuBeta[ee];
	}
	//Also update mu for intercept
	double denom = SigmaBeta[PP[0]] + KK[0];
	double sumbeta = 0.0;
	for(int ww = 0; ww < KK[0]; ww++){
		sumbeta += intercept[ww];
	}
	double tempmu = (sumbeta + SigmaBeta[PP[0]]*MuBeta[PP[0]])/denom;
	double tempssq = SigmaBeta[PP[0]]/denom;
	MuBeta[PP[0]] = rnorm(tempmu,tempssq);
	postMu[ii+niter[0]*PP[0]] = MuBeta[PP[0]];
//Update variance for slopes
	for(int ee = 0; ee < PP[0]; ee++){
 	double B = 0.0;
		for(int vv = 0;vv<KK[0];vv++){
			B += square(beta[ee+vv*PP[0]] - MuBeta[ee]);
		}
		B = B/2.0 + PriorB[0];
		double A = PriorA[0] + KK[0]/2.0;
		SigmaBeta[ee] = 1.0/rgamma(A,1/B);
		postVar[ii+niter[0]*ee] = SigmaBeta[ee];
	}
	//Also update variance for intercept
	double B = 0.0;
	for(int vv = 0; vv < KK[0]; vv++){
		B += square(intercept[vv] - MuBeta[PP[0]]);
	}
	B = B/2.0 + PriorB[0];
	double A = PriorA[0] + KK[0]/2.0;
	SigmaBeta[PP[0]] = 1.0/rgamma(A,1/B);	
	postVar[ii+niter[0]*PP[0]] = SigmaBeta[PP[0]];

	    //Update varaince for ZZ    	
	for(int vv = 0; vv < dd[0]; vv++){	
		D[vv] = D[vv]/2.0 + PriorB[0];
		C = PriorA[0] + (sumAll)/2.0;
		VarZ[vv] = 1.0/rgamma(C,1.0/D[vv]);
	}	
	Zvar1[ii] = VarZ[0];
	Zvar2[ii] = VarZ[1];
	likelihood[ii] = llikall;
	PutRNGstate();
	}
	delete[] accbeta;
	delete[] tuneBeta;	
	delete[] sumn;
	delete[] D;
		 }


} 


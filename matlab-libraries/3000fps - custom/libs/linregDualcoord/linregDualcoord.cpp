#include <math.h>
#include <float.h>
#include <mex.h>
#include <omp.h>

int **compressFeatures(char *matFeatures, int numInst, int numFeat, int iTestB, int iTestE){
    int **ret = new int *[numFeat];
    for(int i=0;i<numFeat;i++){
        char *iFVec = &matFeatures[i*numInst];
        
        // Allocate compressed feature
        int nInst = 0;
        for(int j=0;j<iTestB;j++){ if(iFVec[j]==1) nInst++; }
        for(int j=iTestE;j<numInst;j++){ if(iFVec[j]==1) nInst++; }
        ret[i] = new int[nInst+1];
        ret[i][nInst] = -1;
        
        // Compress feature
        int cInst = 0;
        for(int j=0;j<iTestB;j++){ if(iFVec[j]==1) ret[i][cInst++] = j; };
        for(int j=iTestE;j<numInst;j++){ if(iFVec[j]==1) ret[i][cInst++] = j; };
    }
    
    return ret;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    mwSize sizes3[3];
    int *tsizes;
    
    // Check there are enough input variables
    if(nrhs < 3) mexErrMsgTxt("The function requires three parameters: An NxF array of N samples and F features, an NxT array with N samples and T targets, and the regularization weight.");
    
    // Read features matrix
    tsizes = (int *)mxGetDimensions(prhs[0]);
    if(mxGetClassID(prhs[0]) != mxINT8_CLASS || mxGetNumberOfDimensions(prhs[0]) > 2) mexErrMsgTxt("Feature matrix must be an NxF UINT8 binary-valued array of N samples and F features.");
    char *      matFeatures = (char *) mxGetPr(prhs[0]);
    int         numInst     = (int) tsizes[0];
    int         numFeat     = (int) tsizes[1];
    
    // Read targets matrix
    tsizes = (int *)mxGetDimensions(prhs[1]);
    if(mxGetClassID(prhs[1]) != mxSINGLE_CLASS || mxGetNumberOfDimensions(prhs[1]) > 2) mexErrMsgTxt("Targets matrix must be an NxT single precision array of N samples and T targets.");
    if(tsizes[0] != numInst) mexErrMsgTxt("Targets matrix must have as many instances (rows) as the feature matrix.");
    float *     matTargets  = (float *) mxGetPr(prhs[1]);
    int         numTarg     = (int) tsizes[1];
    
    // Read lambda weight
    tsizes = (int *)mxGetDimensions(prhs[2]);
    if(mxGetClassID(prhs[2]) != mxSINGLE_CLASS || mxGetNumberOfDimensions(prhs[2]) > 2) mexErrMsgTxt("Regularization weight must be a single precision scalar.");
    float       valLambda   = ((float *)mxGetPr(prhs[2]))[0];
    float       valNLambda  = numInst * valLambda;
    
    // Read number of folds (if specified)
    int numFolds = 1;
    if(nrhs >= 4){
        tsizes = (int *)mxGetDimensions(prhs[3]);
        if(mxGetClassID(prhs[3]) != mxINT32_CLASS || mxGetNumberOfDimensions(prhs[3]) > 2) mexErrMsgTxt("Number of folds must be a int32 positive value.");
        if(tsizes[0] > 1 || tsizes[1] > 1) mexErrMsgTxt("Number of folds must be a positive int32 value.");
        numFolds = *(int *)mxGetPr(prhs[3]);
        if(numFolds < 1 || numFolds > numInst) mexErrMsgTxt("Number of folds must be a scaler between 1 and N, being N the number of instances.");
    }
    
    // Prepare weights matrix
    sizes3[0] = numFeat; sizes3[1] = numTarg; sizes3[2] = numFolds;
    plhs[0] = mxCreateNumericArray(3, sizes3, mxSINGLE_CLASS, mxREAL);
    float  *matWeights  = (float *)mxGetPr(plhs[0]);
    long    sMatWeights = numFeat * numTarg;
    
    // Prepare validation results matrix
    plhs[1] = mxCreateNumericMatrix(numInst, numTarg, mxSINGLE_CLASS, mxREAL);
    float *matResTest   = (float *)mxGetPr(plhs[1]);
    
    // Allocate other temporary matrices
    long    sMatResults = numInst * numTarg;
    float *matResults   = new float[numFolds*sMatResults];
    float *matDotp      = new float[numFeat];
    
    // Prepare individual features dot-product + valNLambda
    for(int iF=0;iF<numFeat;iF++){
        char *iFVec = &matFeatures[iF*numInst];
        matDotp[iF] = valNLambda;
        for(int i=0;i<numInst;i++){
            if(iFVec[i] == 1) matDotp[iF]++;
        }
        matDotp[iF] = 1 / matDotp[iF];
    }
    
    // Get partition size
    int partSize = (numFolds == 1) ? 0 : numInst / numFolds;
    
    #pragma omp parallel for
    for(int iFld=0;iFld<numFolds;iFld++){
        int iTestB = partSize*iFld;
        int iTestE = iTestB+partSize;
        
        // Define features, weights and results matrices
        int    **compFeats = compressFeatures(matFeatures, numInst, numFeat, iTestB, iTestE);
        float   *mWeights  = &matWeights[iFld*sMatWeights];
        float   *mResults  = &matResults[iFld*sMatResults];
        for(int i=0;i<sMatResults;i++) mResults[i] = matTargets[i];
        
        // Start dual coordinate descent
        int itNum = 0;
        float cerror = FLT_MAX;
        while(itNum<1000){
            // Minimize each feature weights sequentially
            for(int iF=0;iF<numFeat;iF++){
                int *iFVec = compFeats[iF];
                float *iWgts = &mWeights[iF];

                for(int iT=0;iT<numTarg;iT++){
                    float *tTarg = &mResults[iT*numInst];
                    float owght  = iWgts[iT*numFeat];
                    float nwght  = 0;

                    // Add current contribution of the feature to target & recalculate weight
                    for(int i=0;iFVec[i]!=-1;i++){
                        tTarg[iFVec[i]] += owght;
                        nwght += matDotp[iF]*tTarg[iFVec[i]];
                    }

                    // Subtract new contribution of the feature from target & save new weight
                    for(int i=0;iFVec[i]!=-1;i++) tTarg[iFVec[i]] -= nwght;
                    iWgts[iT*numFeat] = nwght;
                }
            }

            // Calculate new error
            float nerror = 0;
            for(int iT=0;iT<numTarg;iT++){
                float *tTarg = &mResults[iT*numInst];
                float terror = 0;
                for(int iI=0;iI<numInst;iI++) terror += tTarg[iI]*tTarg[iI];
                nerror += sqrt(terror);
            }

            if(nerror >= cerror) break;
            cerror = nerror;
            itNum++;
        }
        
        // Perform linear regression to obtain validation results
        if(nlhs > 1 && numFolds > 1){
            for(int i=iTestB;i<iTestE;i++){
                char *iFVec = &matFeatures[i];
                for(int j=0;j<numTarg;j++){
                    float *iWgts = &mWeights[j*numFeat];
                    int idx = i+j*numInst;
                    matResTest[idx] = 0;
                    for(int k=0;k<numFeat;k++){ if(iFVec[k*numInst] == 1) matResTest[idx] += iWgts[k]; }
                }
            }
        }
        
        for(int i=0;i<numFeat;i++) delete[] compFeats[i];
        delete[] compFeats;
        mexPrintf("Converged to a solution! (%d steps)\n", itNum+1);
    }
    
    // Perform linear regression to obtain results if single partition
    if(nlhs > 1 && numFolds == 1){
        for(int i=0;i<numInst*numTarg;i++){
            matResTest[i] = matTargets[i] - matResults[i];
        }
    }
    
    delete[] matResults;
    delete[] matDotp;
}
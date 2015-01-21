#include <math.h>
#include <float.h>
#include <windows.h>
#include <process.h>
#include <mex.h>

float   *minimSpace;
float   *glMeans;
float   *pixValues;
int      minimDims;
int      numSamples;

#define NUM_THREADS     8
#define MSG_NEWTASK     (WM_APP + 1)
#define POW2(x)         ((x)*(x))

struct ThreadData{
    int        iP1, iP2;
    float      spVar;
    
    ThreadData(){}
    ThreadData(int iP1, int iP2) : iP1(iP1), iP2(iP2), spVar(0){};
};

// ***********************************************************
// ** SPLITTING METHODS
// ***********************************************************

inline float calcInterVariance(int spSizeL, float *spMeansL, int spSizeR, float *spMeansR){
    float d1=0, d2=0;
    for(int i=0;i<minimDims;i++){
        d1 += POW2(spMeansL[i] - glMeans[i]);
        d2 += POW2(spMeansR[i] - glMeans[i]);
    }
    return (spSizeL*sqrt(d1) + spSizeR*sqrt(d2)) / 2;
}

unsigned int evaluatePixelPair(void *lpParam){
    float * spMeansL    = new float[minimDims];
    float * spMeansR    = new float[minimDims];
    MSG msg;
    
    // Initialize thread queue & report to parent thread
    PeekMessage(&msg, NULL, WM_USER, WM_USER, PM_NOREMOVE);
    SetEvent(*(HANDLE *)lpParam);
    
    // Process incoming messages
    while(GetMessage(&msg, NULL, 0, 0) != 0){
        if(msg.message != MSG_NEWTASK) continue;
        ThreadData *data = (ThreadData *)msg.wParam;

        // ** Find best split and store result
        // **************************************************

        // Initialize means
        for(int i=0;i<minimDims;i++){
            spMeansL[i] = 0;
            spMeansR[i] = 0;
        }
        
        // Acccumulate target space for split value = 0
        int nSampL = 0, nSampR = 0;
        float * pix1 = &pixValues[numSamples * data->iP1];
        float * pix2 = &pixValues[numSamples * data->iP2];
        for(int i=0;i<numSamples;i++){
            if(pix1[i]<pix2[i]){
                nSampL++;
                for(long j=0;j<minimDims;j++) spMeansL[j] += minimSpace[j*numSamples+i];
            }else{
                nSampR++;
                for(long j=0;j<minimDims;j++) spMeansR[j] += minimSpace[j*numSamples+i];
            }
        }
        
        // Calculate means
        for(int i=0;i<minimDims;i++){
            spMeansL[i] = spMeansL[i] / nSampL;
            spMeansR[i] = spMeansR[i] / nSampR;
        }
        
        // Calculate split variance
        data->spVar = calcInterVariance(nSampL, spMeansL, nSampR, spMeansR);
        
        // ** Inform task is complete
        // **************************************************
        
        SetEvent(*(HANDLE *)lpParam);
    }
    
    delete[] spMeansL, spMeansR;
    return 0;
}

// ***********************************************************
// ** MAIN FUNCTION
// ***********************************************************

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    int *tsizes;
    
    // ** PERFORM DATA CHECKS & OBTAIN MATRIX DIMENSIONS
    // ******************************************************
    
    // Check there are enough input variables
    if(nrhs < 2) mexErrMsgTxt("The function requires at least two parameters: An NxU array of N samples and U split spaces, and an NxV array with N samples and V minimization spaces.");
    
    // Read split spaces
    tsizes = (int *)mxGetDimensions(prhs[0]);
    if(mxGetClassID(prhs[0]) != mxSINGLE_CLASS || mxGetNumberOfDimensions(prhs[0]) > 2) mexErrMsgTxt("Split space must be an NxU single precision array of N samples and U split spaces.");
    pixValues      = (float *) mxGetPr(prhs[0]);
    int numPixels  = (int) tsizes[1];
    numSamples     = (int) tsizes[0];
    if(numSamples <= 1) mexErrMsgTxt("At least two samples are required to define a splitting.");
    
    // Read variance minimization space
    tsizes = (int *)mxGetDimensions(prhs[1]);
    if(mxGetClassID(prhs[1]) != mxSINGLE_CLASS || mxGetNumberOfDimensions(prhs[1]) > 2) mexErrMsgTxt("Minimization space must be an NxV single precision array.");
    if(tsizes[0] != numSamples) mexErrMsgTxt("Minimization space must be an NxV array with as many samples as the split space.");
    minimSpace     = (float *) mxGetPr(prhs[1]);
    minimDims      = tsizes[1];
    
    // Read number of pairs to return, if parameter set
    int numResults = 1;
    if(nrhs >= 3){
        tsizes = (int *)mxGetDimensions(prhs[2]);
        if(mxGetClassID(prhs[2]) != mxSINGLE_CLASS || mxGetNumberOfDimensions(prhs[2]) > 2) mexErrMsgTxt("Number of outputs must be a single-precision scalar value.");
        if(tsizes[0] > 1 || tsizes[1] > 1) mexErrMsgTxt("Number of outputs must be a single-precision scalar value.");
        numResults = (int)(*(float *)mxGetPr(prhs[2]));
        if(numResults <= 0 || numResults > numPixels*(numPixels-1)/2) mexErrMsgTxt("Number of results must be a positive scalar equal or smaller than Ux(U-1)/2, being U is the number of split spaces.");
    }

    // ** ALLOCATE RETURN VARIABLES
    // ******************************************************
    
    // Prepare return matrix
    plhs[0]     = mxCreateNumericMatrix(numResults, 1, mxSINGLE_CLASS, mxREAL);
    plhs[1]     = mxCreateNumericMatrix(numResults, 1, mxSINGLE_CLASS, mxREAL);
    plhs[2]     = mxCreateNumericMatrix(numResults, 1, mxSINGLE_CLASS, mxREAL);
    float *ri1 = (float *)mxGetPr(plhs[0]);
    float *ri2 = (float *)mxGetPr(plhs[1]);
    float *var = (float *)mxGetPr(plhs[2]);
    
    // ** PRE-CALCULATE GLOBAL MEANS OF TARGET VARIABLES
    // ******************************************************
    
    // Calculate global means
    glMeans = new float[minimDims];
    for(int i=0;i<minimDims;i++){
        float *minimSp = &minimSpace[i*numSamples];
        
        glMeans[i] = 0;
        for(int j=0;j<numSamples;j++) glMeans[i] += minimSp[j];
        glMeans[i] /= numSamples;
    }
    
    // ** LAUNCH TASKS
    // ******************************************************
    
    // Prepare thread management variables
    int numTasks                 = (numPixels-1)*numPixels/2;
    ThreadData *   tasksData     = new ThreadData[numTasks];
    HANDLE *       threadHandles = new HANDLE[NUM_THREADS];
    ThreadData **  threadsData   = new ThreadData *[NUM_THREADS];
    unsigned int * threadIds     = new unsigned int[NUM_THREADS];
    
    // Prepare work packets
    int idx = 0;
    for(int i=0;i<numPixels-1;i++){
        for(int j=i+1;j<numPixels;j++){
            tasksData[idx] = ThreadData(i, j);
            idx++; 
        }
    }
    
    // Prepare ready state events
    HANDLE *evtReadyState = new HANDLE[NUM_THREADS];
    for(int i=0;i<NUM_THREADS;i++) evtReadyState[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
    
    // Launch worker threads & wait their initialization
    for(int i=0;i<NUM_THREADS;i++) threadHandles[i] = (HANDLE) _beginthreadex(NULL, 0, evaluatePixelPair, &evtReadyState[i], 0, &threadIds[i]);
    WaitForMultipleObjects(NUM_THREADS, evtReadyState, TRUE, INFINITE);
    
    // Initialize results variables
    ThreadData **thrBest = new ThreadData *[numResults];
    for(int i=0;i<numResults;i++) var[i] = 0;
    
    // Launch initial tasks
    for(int i=0;i<NUM_THREADS;i++){
        threadsData[i] = &tasksData[i];
        PostThreadMessage(threadIds[i], MSG_NEWTASK, (WPARAM)threadsData[i], 0);
    }
    
    // Manage threads
    int numRunning = NUM_THREADS;
    int started = NUM_THREADS, finished = 0;
    while(finished<numTasks){
        int iThrd = (int)WaitForMultipleObjects(numRunning, evtReadyState, FALSE, INFINITE);
        finished++;
        
        // Check results of finished thread
        int iMinV = 0;
        for(int i=1;i<numResults;i++){ if(var[i]<var[iMinV]) iMinV = i; }
        if(threadsData[iThrd]->spVar > var[iMinV]){
            var[iMinV]     = threadsData[iThrd]->spVar;
            thrBest[iMinV] = threadsData[iThrd];
        }
        
        // Launch new thread if necessary
        if(started < numTasks){
            threadsData[iThrd] = &tasksData[started];
            PostThreadMessage(threadIds[iThrd], MSG_NEWTASK, (WPARAM)threadsData[iThrd], 0);
            started++;
        }else{
            evtReadyState[iThrd] = evtReadyState[--numRunning];
        }
    }
    
    // Close threads
    for(int i=0;i<NUM_THREADS;i++){
        PostThreadMessage(threadIds[i], WM_QUIT, 0, 0);
        CloseHandle(threadHandles[i]);
    }
    
    // Sort results by variance
    for(int i=1;i<numResults;i++){
        for(int j=i;j>0;j--){
            if(var[j]<=var[j-1]) break;
            
            ThreadData *tmpD = thrBest[j];
            thrBest[j]   = thrBest[j-1];
            thrBest[j-1] = tmpD;
            
            float tmpV = var[j];
            var[j] = var[j-1];
            var[j-1] = tmpV;
        }
    }

    // Assign best results
    for(int i=0;i<numResults;i++){
        ri1[i] = thrBest[i]->iP1+1;
        ri2[i] = thrBest[i]->iP2+1;
    }
    
    delete[] glMeans;
    delete[] threadHandles, threadsData, threadIds;
    delete[] evtReadyState;
    delete[] tasksData;
    delete[] thrBest;
}
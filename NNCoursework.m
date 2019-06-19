%% SECTION 1: DATA EXPLORATION/PREPARATION AND VARIABLE INITIALISATION
%==================================================================================================%

WindSpeed = readtable('WindSpeed.csv'); % Import the data: a Wind Speed Time Series from 1994 to 2015.

% NORMALISE THE TIME SERIES
maxWS = max(WindSpeed{:,2}); % Extract max wind speed
minWS = min(WindSpeed{:,2}); % Extract min wind speed
WindSpeed{:,2} = (WindSpeed{:,2}-minWS)/(maxWS-minWS); % Normalise

% VARIABLE INITIALISATION 

ID = 2:1:10; % Order of model for SVM and MLP

% Variables for SVM (stored for each order to be used for visualisations)

HyperParamsSVM = []; % Each Combination of hyperparameters will be appended to this variable on each run
dataTestOutputSVM = cell(1,10); % For storing the actual SVM test output data
dataTrainOutputSVM = cell(1,10); % For storing the actual SVM training output data
PredictionsTestSVM = cell(1,10); % For storing the Test Predictions
PredictionsTrainSVM = cell(1,10); % For storing the Training Predictions
DiffSVM = cell(1,10); % For storing SVM test residuals (used for histogram of residuals) 
mseSVMTrain = cell(1,10); % For storing the SVM Training MSE for the optimized model 
mseSVMTest = cell(1,10); % For storing the SVM Test MSE for the optimized model
SVMMdl = cell(1,10); % For storing the optimized SVM model for each order

% Variables for MLP (stored for each order to be used for visualisations)

HyperParamsMLP = []; % Each Combination of hyperparameters will be appended to this variable on each run
dataTestOutputMLP = cell(1,10); % For storing the actual MLP test output data
PredictionsMLP = cell(1,10); % For storing the Test Predictions
DiffMLP = cell(1,10); % For storing the MLP test residuals (used for histogram of residuals)
mseMLPTest = cell(1,10); % For storing the MLP Test MSE for the optimized model
mseMLPTrain = cell(1,10); % For storing the MLP Training MSE for the optimized model
MLPMdl = cell(1,10); % For storing the optimized MLP model for each order
MLPtr = cell(1,10); % For storing training information for each model

%% SECTION 2: MODEL SELECTION, TRAINING AND EVALUATION 
%==================================================================================================%

% A for loop is run for each order d from 2 to 10.

% Inside the for loop is the model selection for both SVM and MLP followed
% using Grid Search and 10-fold Cross Validation.

% A final model is trained for each order with the best hyperparameters and
% the MSE and Residuals are calculated from the Test data.

for d = ID
    
    % The data set is split up so there are rows of d input and 1 output.
    
    X = [];
    for i = 1:d:7744
        j = i+d;
        X = [X; WindSpeed{i:j,2}'];
    end

    X = array2table(X);
    
    % The data is partitioned using Holdout so that 80% is used for
    % training & model selection and 20% is held back for testing.

    cvpt = cvpartition(X{:,1},'Holdout',0.2);
    dataTrain = X(training(cvpt),:);
    dataTest = X(test(cvpt),:);
    
    % SVM MODEL SELECTION, TRAINING & TESTING
    
    % Variables to be used in the grid search are initialised below.
    
    Kernel = {'linear', 'gaussian', 'polynomial'}; 
    PolyOrd = [2,3,4]; % due to computational time and a bad fit, polynomial order was limited at 4
    logC = [-3,-2,-1,0,0.5,1,1.5,2,2.5,3]; % Box Constraint ranges from e^-3 to e^3
    Epsilon = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]; % Epsilon ranges from 0.01 to 0.5.
    
    % GRID SEARCH (runs through every combination of hyperparameters stated
    % above, computes 10-fold cross validation MSE and stores all
    % hyperparamters and corresponding mse in an array for later use.
    % The training data is used for 10-fold cross validation.

    for k = Kernel
        if strcmp(k,'polynomial') == 1 % Polynomial has an extra hyperparamter in polynomial order so loop is seperate
            for p = PolyOrd
                for e = Epsilon
                    for C = logC
                        Mdl = fitrsvm(dataTrain(:,1:d),dataTrain(:,d+1),'KernelFunction',char(k),'PolynomialOrder',p,'Epsilon',e,'BoxConstraint',exp(C),'Kfold',10);
                        mse = kfoldLoss(Mdl);
                        T = [d k p e C mse];
                        disp(T) % Can track how training is going.
                        HyperParamsSVM = [HyperParamsSVM; T]; % Appending new hyperparameters to array.
                    end
                end
            end
        else
            for e = Epsilon
                for C = logC
                    Mdl = fitrsvm(dataTrain(:,1:d),dataTrain(:,d+1),'KernelFunction',char(k),'Epsilon',e,'BoxConstraint',exp(C),'Kfold',10);
                    mse = kfoldLoss(Mdl);
                    T = [d k 'N/A' e C mse];
                    disp(T) % Can track how training is going.
                    HyperParamsSVM = [HyperParamsSVM; T]; % Appending new hyperparameters to array.
                end
            end
        end
    end
    
    % Finding the best hyperparameters for the current order.
    
    f = find([HyperParamsSVM{:,1}] ==  d);
    
    H = HyperParamsSVM(f,:);
    
    [M,I] = min(cell2mat(H(:,6)));
    
    % Training the SVM model on all the training data using the best
    % hyperparameters.

    if strcmp(H{I,2},'polynomial') == 1
        SVMMdl{d} = fitrsvm(dataTrain(:,1:d),dataTrain(:,d+1),'KernelFunction',H{I,2},'PolynomialOrder',H{I,3},'Epsilon',H{I,4},'BoxConstraint',exp(H{I,5}));
    else
        SVMMdl{d} = fitrsvm(dataTrain(:,1:d),dataTrain(:,d+1),'KernelFunction',H{I,2},'Epsilon',H{I,4},'BoxConstraint',exp(HyperParamsSVM{I,5}));
    end
    
    dataTrainOutputSVM{d} = dataTrain(:,d+1); % store training data outputs in cell
    PredictionsTrainSVM{d} = predict(SVMMdl{d},dataTrain(:,1:d)); % Calculate training data predictions and store in cell
    mseSVMTrain{d} = immse(table2array(dataTrainOutputSVM{d}),table2array(PredictionsTrainSVM(d))); % calculate the Training MSE and store in cell
  
    dataTestOutputSVM{d} = dataTest(:,d+1); % store test data outputs in cell
    PredictionsTestSVM{d} = predict(SVMMdl{d},dataTest(:,1:d)); % calculate test predictions and store in cell

    DiffSVM{d} = table2array(dataTestOutputSVM{d})-table2array(PredictionsTestSVM(d)); % calculate test residuals and store in cell
    mseSVMTest{d} = immse(table2array(dataTestOutputSVM{d}),table2array(PredictionsTestSVM(d))); % calculate the Test MSE and sotre in cell
    % MLP MODEL SELECTION, TRAINING & TESTING
    
    % Variables to be used in the grid search are initialised below.
    
    trainFcn = 'trainlm'; % trainlm used as vastly increases computational speed
    HiddenLayers = 5:5:100; % hidden layers from 5 - 100
    LearningRate = 0.001:0.05:1; % learning rate from 0.001 to 1
    
    % GRID SEARCH (runs through every combination of hyperparameters stated
    % above, computes 10-fold cross validation MSE and stores all
    % hyperparamters and corresponding mse in an array for later use.
    % The training data is used for 10-fold cross validation.
    
    for h = HiddenLayers
        for l = LearningRate
            
            % net initialised with h hidden layers trainlm for training
            % function and l for learning rate.

            net = fitnet(h,trainFcn);
            net.trainParam.showWindow = false;
            net.trainParam.mu = l;
            
            % Choose Input and Output Pre/Post-Processing Functions
            
            net.input.processFcns = {'removeconstantrows','mapminmax'};
            net.output.processFcns = {'removeconstantrows','mapminmax'};

            % Choose a Performance Function
     
            net.performFcn = 'mse';  % Mean Squared Error
            
            %Split Data for 10-fold Cross-Validation
            
            cvpt = cvpartition(size(dataTrain,1),'KFold',10);
            valPerf = zeros(1,10); % To store MSE for each fold (will take mean for final validation MSE).
            
            % Run 10-fold cross validation
            
            for i = 1:cvpt.NumTestSets
                trIdx = cvpt.training(i); % training index
                valIdx = cvpt.test(i); % validation index
                
                % Train the Network on training index data
                [net,tr] = train(net,cell2mat(num2cell(transpose(table2array(dataTrain(trIdx,1:d))))),cell2mat(num2cell(transpose(table2array(dataTrain(trIdx,d+1))))));
                
                % Calculate MSE for this fold
                y = net(cell2mat(num2cell(transpose(table2array(dataTrain(valIdx,1:d))))));
                valPerf(i) = perform(net,cell2mat(num2cell(transpose(table2array(dataTrain(valIdx,d+1))))),y);
            end
            mse = mean(valPerf); % Take mean for final validation MSE
            T = [d h l mse];
            disp(T) % Can track how training is going.
            HyperParamsMLP = [HyperParamsMLP; T];  % Appending new hyperparameters to array.
        end
    end 
    
    % find best hyperparameters for this order.
    
    f = find([HyperParamsMLP(:,1)] ==  d);
    
    H = HyperParamsMLP(f,:);
    
    [M,I] = min(H(:,4));
    
    % Train model using best hyperparameters and full set of training data.
    
    net = fitnet(H(I,2),trainFcn);
    net.trainParam.mu = H(I,3);
    
    % Choose Input and Output Pre/Post-Processing Functions
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};
    
    % Choose a Performance Function
    net.performFcn = 'mse';  % Mean Squared Error
    
    % Train MLP and store in cell
    [MLPMdl{d},MLPtr{d}] = train(net,cell2mat(num2cell(transpose(table2array(dataTrain(:,1:d))))),cell2mat(num2cell(transpose(table2array(dataTrain(:,d+1))))));
    
    % Calculate test performance and store in cell
    y = MLPMdl{d}(cell2mat(num2cell(transpose(table2array(dataTest(:,1:d))))));
    mseMLPTest{d} = perform(MLPMdl{d},cell2mat(num2cell(transpose(table2array(dataTest(:,d+1))))),y);
    
    % Calculate training performancec and store in cell
    y = MLPMdl{d}(cell2mat(num2cell(transpose(table2array(dataTrain(:,1:d))))));
    mseMLPTrain{d} = perform(MLPMdl{d},cell2mat(num2cell(transpose(table2array(dataTrain(:,d+1))))),y);
    
    % store actual test output in cell
    dataTestOutputMLP{d} = dataTest(:,d+1);
    
    % store test predictions in cell
    PredictionsMLP{d} = MLPMdl{d}(cell2mat(num2cell(transpose(table2array(dataTest(:,1:d))))));
    
    % Calculate residuals and store in cell
    DiffMLP{d} = table2array(dataTestOutputMLP{d})-transpose(table2array(PredictionsMLP(d)));
end

%% SECTION 3: VISUALISATIONS FOR REPORT 
%==================================================================================================%

% Line plot of Test MSE for SVM and MLP on y-axis against order on x-axis

figure;
plot(ID,cell2mat(mseSVMTest),'b',ID,cell2mat(mseMLPTest),'r')
xlabel('Order')
ylabel('MSE')
title('Test MSE for MLP and SVM')
legend('SVM Test MSE','MLP Test MSE')
saveas(gcf,'MSE_Order.eps','epsc')

% Time series plot of Normalized wind speed.

figure;
plot(WindSpeed{:,1},WindSpeed{:,2});
xlabel('Date')
ylabel('Normalized Wind Speed (m/s)')
title('Normalized Wind Speed Time Series')
saveas(gcf,'WS_TimeSeries.eps','epsc')

% Histogram of normalized wind speeds to show how values are distributed
% between 0 and 1.

figure;
hist(WindSpeed{:,2})
xlabel('Normalized Wind Speed (m/s)')
ylabel('Frequency')
title('Distribution of Normalize Wind Speeds')
saveas(gcf,'WS_Distribution.eps','epsc')

% Histogram of residuals for SVM and MLP

figure;
subplot(2,1,1)
h1 = histogram(cell2mat(DiffSVM(9)));
h1.NumBins = 10;
h1.FaceColor = 'blue';
xlabel('Residuals')
ylabel('frequency')
title('Residuals for SVM')
subplot(2,1,2)
h2 =  histogram(cell2mat(DiffMLP(9)));
h2.NumBins = 10;
h2.FaceColor = 'red';
xlabel('Residuals')
ylabel('frequency')
title('Residuals for MLP')
saveas(gcf,'Residuals.eps','epsc')

% Extract mse from HyperParamsMLP so that Grid Search can be visualised as
% a surface. The optimized hyperparameter for Order is 9 so that is used
% for this visualisation.

f = find([HyperParamsMLP(:,1)] ==  9);
H = HyperParamsMLP(f,:);
Surface = zeros(20,20);

for i = 1:20
    for j = 1:20
        f1 = find([H(:,2) == HiddenLayers(i) & H(:,3) == LearningRate(j)]);
        Surface(j,i) = H(f1,4);
    end
end

% Plot pcolor grid to show MLP Grid Search.

figure;
pcolor(HiddenLayers,LearningRate,Surface)
xlabel('Number of Hidden Layers')
ylabel('Initial Learning Rate')
title('Grid Search Hyperparameter Optimisation of MLP')
saveas(gcf,'MLP_Hyperparameters.eps','epsc')

% Extract mse from HyperParamsSVM for a liner, guassian and polynomial kernel 
% so that Grid Search can be visualised as a surface.
% The optimized hyperparameter for Order is 9 so that is used
% for this visualisation.

f = find([HyperParamsSVM{:,1}] ==  9 );
H = HyperParamsSVM(f,:);
flinear = find(contains(H(:,2),'linear'));
HLinear = H(flinear,:);
fgaussian = find(contains(H(:,2),'gaussian'));
HGaussian = H(fgaussian,:);
fpolynomial = find(contains(H(:,2),'polynomial'));
HPolynomial = H(fpolynomial,:);
SurfaceLinear = zeros(10,11);
SurfaceGaussian = zeros(10,11);
SurfacePolynomial = zeros(10,11);

for i = 1:10
    for j = 1:11
        f2 = find([cell2mat(HLinear(:,4)) == Epsilon(j) & cell2mat(HLinear(:,5)) == logC(i)]);
        SurfaceLinear(i,j) = cell2mat(HLinear(f2,6));
        f3 = find([cell2mat(HGaussian(:,4)) == Epsilon(j) & cell2mat(HGaussian(:,5)) == logC(i)]);
        SurfaceGaussian(i,j) = cell2mat(HGaussian(f3,6));
    end
end

% Plot two 3-D surfaces, one representing the grid search for the Linear Kernel
% and the other representing the grid search for the Gaussian Kernel.

figure;
subplot(2,1,1)
s1 = surf(Epsilon,logC,SurfaceLinear,'FaceAlpha',0.5);
s1.EdgeColor = 'none';
ylabel('log(BoxConstraint)')
xlabel('Epsilon')
zlabel('MSE')
title('Surface of Hyperparamter Optimization for SVM (Linear Kernel)')
subplot(2,1,2)
s2 = surf(Epsilon,logC,SurfaceGaussian,'FaceAlpha',0.5);
s2.EdgeColor = 'none';
ylabel('log(BoxConstraint)')
xlabel('Epsilon')
zlabel('MSE')
title('Surface of Hyperparamter Optimization for SVM (Gaussian Kernel)')
saveas(gcf,'SVM_Hyperparameters.eps','epsc')








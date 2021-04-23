# My Cheat-sheet

# AWS Infra

## AWS Services

* `Polly` --> Text to speech
* `Comprehend` --> NLP --> extract relationships and metadata
* `lex` --> Chatbots, speech to text, text to text (does not speak!)
* `Transcribe` --> Speech to text (not for chatbots; does not recognize intend)
* `Translate` --> Language translation
* `Textract` --> OCR
* `Rekognition` --> Face Sentiment, Face Search in photos or Video, Searchable Video Lib, Moderation
* `Forecast` --> Managed Service for time Series forecasting
* `Personalise` --> Creates high-quality recommendations for your websites and applications

## SageMaker
* `SageMaker` --> A fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly.
* `Hyperparameter Tuning`:
  * Define Metrics
  * Define Hyperparameter Ranges
  * Run Random or Bayesian Tuning
  * API --> `HyperparameterTuner()`
* `Batch Transform`:
  * Pre-process datasets to remove noise or bias that interferes with training or inference from your dataset.
  * Get inferences from large datasets.
  * Run inference when you don't need a persistent endpoint.
* `Neo` --> Enables machine learning models to train once and run anywhere in the cloud and at the edge
* `Inference Pipelines` --> An Amazon SageMaker model that is composed of a linear sequence of two to five containers that process requests for inferences on data
* `Elastic Inference` --> 
  * Speed up the throughput and decrease the latency of getting real-time inferences from your deep learning models
  * Use a private link end-point service
* `Standard Scaler` --> Normalise Columns
* `Nomaliser` --> Normalise Rows 
* `DescribeJob API` --> To check what went wrong
* `Deploying a model`:
  * Create a model
  * Create an endpoint configuration for an HTTPS endpoint
  * Create an HTTPS endpoint

## Streaming Services
* `Kinesis Data Stream` --> Stream large volumes fo data for processing (EMR, Lambda, KDA)
* `Kinesis Video Stream` --> Stream Videos for Analytics, ML or video processing
* `Kinesis Firehose` --> Stream data into an end point (S3, ES, Splunk)
* `Kinesis Data Analytics (KDA)` --> Apply analytics on Data (Java libs (Flink), SQL)

## Other Services
* `Amazon Fsx` --> High-performance file system (cost effective too).
* `S3` --> Simple Storage Service: object storage service (structure or unstructured) that offers scalability, data availability, security, and performance (at low cost).
* `Glue` --> A serverless ETL service that crawls your data, builds a data catalog, performs data preparation, data transformation, and data ingestion
* `CloudWatch` --> A monitoring and observability service that provides you with data and actionable insights to monitor your applications (logs).
* `CloudTrail` --> Enables governance, compliance, operational auditing, and risk auditing of your AWS account (track API calls).
* `Athena` --> Interactive query service that makes it easy to analyze data in Amazon S3 using standard SQL (needs `AWS Glue`)
* `Amazon EMR` --> Hadoop Env for Spark, Hive etc (not serverless)
* `Lambda` --> Run code without provisioning or managing servers (15 mins life)
* `Step-Functions` --> Lets you coordinate multiple AWS services into serverless workflows (like state machines)
* `AWS Batch`

# ML & Data Science
## ML Lifecycle:
* Data Ingestion
* Data Cleaning
* Feature Selection/Engineering
* Model Training & Optmisation
* Model Performance
* Model Servicing

## Data Processing: Ingestion, Cleaning, Feature Selection & Engineering
* `Pipe\RecordIO` --> Stream data into your ML processing pipeline (faster, due to less I/O)
* Feature Selection: Algorithm runs quicker (**speed up**) and is more effective (**accuracy**).
  * Remove data that has no bearing on the target label: 
    * Look at the correlation of a feature and the target label
    * Look at the feature's variance!
    * Look at percentages of missing data
* Dealing with Missing or imbalanced Data:
  * Imputing new values:
    * Mean of all values (if not too many rows are missing... else remove that row)
    * If a feature has a vary low variance or has many values missing, it can also be removed
  * `Class Imbalance`:
    * Source more data or synthesise more data:
      * `SMOTE` --> Generate mock data using means across existing datapoints (depends on how many you choose)   
    * `Under-sampling` --> Match the records of your smallest class (ignore what's left)
    * `Over-sampling` -->  Replicate minority class to increase it as close to other classes (no variation though...).
    * Try different algorithms!
* Feature Engineering: Engineer new features:
  * E.g. Multiply age with height!
  * Datetime --> Hour of the day
  * `PCA` --> Dimensionality Reduction:
    * Identify centroid (mean value of all points) and move centroid to the center of your axes
    * Draw a minimum bounding box encapsulating all data points
    * The length of the sides of these boxes is a PC and we have as many as the number of features we have
    * The length of each side (PC) defines the compoenent's variance. We can drop the lowest ones.
  * `T-SNE` --> Dimensionality Reduction: "Stochastic Neighbor Embedding":
    * It models each high-dimensional object by a two-or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability.
  * Label encoding/one-hot encoding!
    * For catecorigal features that have no ordinal relationships.
* Splitting and Randomisation:
  * Work so that the distribution of target labels is balanced between testing and training data

## Model Training & Optmisation
* **Gradient Descent**: Used for linear regression, logistic regression and SVMs
  * Defined by:
    * a loss Function
    * a Learning Rate (or step)
  * Optimisers:
    * `Adam` --> Good with escaping Local Minimals
    * `Adagrad` --> Optimise learning Rate
    * `RMSProp` --> Uses Moving Average Gradients  
  * Deep Leanring:
    * Backward and forward propagation!
* **Genetic Algorithms**:
  * Fitness Functions
* **Regularisation**: Apply it when our model overfits!
  * Reduce the model's sensitivity to certain features (e.g. height).
  * Can be done through regression (L1 and L2).
* **Hyperparameters**: External parameters to the training job (e.g. learning rate, epochs, batch size, tree depth, )
  * Hyperparameter Tuning: E.g. Random Bayes (trial and error)!
* **Cross Validation**: 
  * Validation of data can be used to tweak hyperparameters
  * To not lose any training data we use cross-validation: k-fold crioss validation
  * Also useful for comparing algorithms!

## Model Performance
* Confusion Matrix: Used for model performance evaluation (to compare the performance across many algorithms)
   * `Recall/Sensitivity (True Positives Rate)`: $\frac{TP}{TP+FN}$ --> The higher it is the fewer FN we have! (Use-case: Catching Fraud, FN are unacceptable!)
   * `Specificity (True Negatives Rate)`: $\frac{TN}{TN+FP}$ --> The higher it is the fewer FP we have! (Use-case: Content Moderation, FP are unacceptable!)
   * `Precision`: $\frac{TP}{TP+FP}$ --> True positives proportion that where correctly classified.
   * `Accuracy`: $\frac{TP + TN}{ALL}$ --> May imply overfitting if too high!
   * `F1` = $\frac{2*Recall*Precision}{Recall + Precision}$
   * `ROC`: Receiver Operator Curve: Helps with identifying max-specificity and max-sensitivity cut-off points (models).
   * `AUC`: Area Under the Curve: Characterises the overal performance of a model! The larger it is the better!
* Entropy/Gini: Information gain
* Variance: root of squared distances of all points from the mean.
  * Used in evaluating cluster representations
  * Used in PCA

## Some Algorithms:
* Logistical Regression
* Linear Regression
* Support Vector Machines
* Decision Trees
* Random Forests
* K-Means
* K-Nearest Neighbour
* Latent Dirichlet Allocation (LDA) Algorithm

## Image Processing (Algorithms/Models)
* `Semantic Segmentation` --> like Image Classification on Pixels (pixel Colour labelling)
* `Instance Segmentation` --> Identify Objects in a picture or video (people, cars, trees)
* `Image Localisation` --> Identify Main Instance in an Image
* `Image Classification (CNN)` --> Label Images

## NLP (Algorithm/Models)
* Remember bag-of-words, n-grams and padding, lemmatisation, tokenisation, stop-words
* `Word2Vec` --> Maps words to high-quality distributed vectors. Good for sentiment analysis, named entity recognition, machine translation
* `Object2Vec` --> A general-purpose neural embedding algorithm. generalises `Word2Vec`.
* `BlazingText` --> Similar to Word2vec, it provides the Skip-gram and continuous bag-of-words (CBOW) training architectures. Very Scalable though compared to `Word2Vec`.  

## Time Series/Anomaly Detection and more (Algorithms/Models)
* `DeepAR` --> RNN for scalar time series data
* `Random Cut Forests` --> Decition Trees for Anomaly detection (patterns)
* `XGBoost` --> Extrem Gradient boosting applied on Decision Trees (very effective not accounting for deep learning)
* `Factorization Machines Algorithm (FMA)` --> Works with highly dimensional and sparse data inputs

# My Cheat-sheet

## Services

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

## Streaming
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

## Model/Data Processing
* `PCA` --> Dimensionality Reduction
* `T-SNE` --> Dimensionality Reduction
* `Pipe\RecordIO` --> Stream data into your ML processing pipeline (faster, due to less I/O)
* `Optimisers`:
  * `Adam` --> Good with escaping Local Minimals
  * `Adagrad` --> Optimise learning Rate
  * `RMSProp` --> Uses Moving Average Gradients   
* `Class Imbalance`:
  * `SMOTE` --> Generate mock data using means across existing datapoints (depends on how many you choose)
  * `Under-sampling` --> Match the records of your smallest class (ignore what's left)
  * `Over-sampling` -->  Replicate minority class to increase it as close to other classes.
* Confusion Matrix:
   * Recall/Sensitivity: $\frac{TP}{TP+FN}$
   * Specificity: $\frac{TN}{TN+FP}$
   * Precision: $\frac{TP}{TP+FP}$
   * Accuracy: $\frac{TP + TN}{ALL}$
   * F1 = $\frac{2*Recall*Precision}{Recall + Precision}$
   * ROC: Receiver Operator Curve
   * AUC: Area Under the Curve
* Entropy/Gini: Information gain
* Variance: root of squared distances of all points from the mean.
  * Used in evaluating cluster representations
  * Used in PCA

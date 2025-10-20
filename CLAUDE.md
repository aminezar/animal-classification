VUB ML project 2024: Animal Classification
Competition for the Machine Learning course at VUB (2024)

Description
Introduction
For this project, you are required to build a classification model that is able to distinguish 12 classes of animals from images.

Nowadays, a problem like this is almost always solved by using Convolutional Neural Networks (CNNs) directly on the RGB images, and many state-of-the-art implementations exist which classify images as well as humans do, or even better. Another option is the more "old-school" approach of Visual-Bag-of-Words (VBoW) models. In VBoW models, we extract meaningful vector features from images using a technique similar to the Bag-of-Words from Natural Language Processing. This enables us to reduce an image -- with its high-dimensionality from its many RGB pixels -- to a smaller vector that contains the relevant information from that image. This vector can then be used as input to any of our classification algorithms.

For this competition, you have to both build models using VBoW + SciKit-Learn (or XGBoost or a similar non-deep-learning library), but also using convolutional neural networks (CNN), for which you can use any library you prefer (PyTorch, TensorFlow, JAX, etc.). The main requirement is that you have at least a few well-tuned models that are not deep learning, and (at least) one deep learning model. CNNs basically do feature extraction for you, but it's not easy to get them well tuned. In contrast, the traditional approach with VBoW will make you more familiar with commonly used frameworks like SciKit-Learn and OpenCV. Both approaches will get you acquainted with a typical machine learning workflow.

How to start?
You are encouraged to not only use the algorithms seen in the course (although they will provide a solid starting point), but to go beyond this and try out the many other classifiers that are out there. After all, some algorithms perform better on certain tasks than others, and it is up to you to find out which ones are best suited for this particular problem (and why!). You are expected to build a solid machine learning pipeline in which you can rigorously test the different algorithms. To do this, you will need to adhere to the following steps:

1) Problem and data analysis
2) Preprocessing and feature extraction
3) Training/validation/test split
4) Model training and hyperparameter tuning
5) Analysis of performance and errors
6) Learn from analysis and repeat

We advise you to start by checking out the 'startkit' folder, and start building your pipeline from there. Start simple, for example by trying out the Naive Bayes classifier from scikit-learn, or the Logistic Regression classifier from the attached classificiation tutorial, or a (linear) Support Vector Machine (SVM), etc. as a baseline and see how your performance improves compared to the prior probablities "algorithm" we provided in the start_here notebook. Build a strong train/test and (cross-)validation scheme around your classifiers to improve the generalization of your model, and really think about why you are doing this or that. Start checking out everything scikit-learn has to offer (feature extraction, feature selection, other classifiers, performance metrics etc.). If you are not satisfied with what scikit-learn has to offer, build your own solutions (once you progress in the challenge, this might become useful for certain parts). For the convolutional neural networks, start checking some online tutorials on computer vision classification. There are many examples out there that can guide you, or you can also check the official documentation of deep learning frameworks like PyTorch.

Inspect your data thoroughly. Analyze every last bit of information out of your results. Are you using the right performance metrics? Are there outliers? Is your dataset balanced? If not, how do you address this (think about your performance metrics, loss function and the distributions of the train/validation/test splits)? Do you need normalization? If so, are you using the correct normalization? What are the limitations of the dataset? Can you augment your data? Do you need dimensionality reduction? What about feature selection? Do you use regularization? How did you choose your hyperparameters? Are you overfitting? Underfitting? Would having more data benefit your model? Can you quantify that with a learning curve? Is there information-leakage from your training set to your test/validation set? Are your features informative enough? Is your model powerful enough? Should you try an ensemble of models? Do you pick a linear model, or a non-linear one? Why? How could you improve the model's weak spots? Why is your model misclassifying these samples? Et cetera, et cetera.

Some notes on using CNNs
You can use any framework, well-known examples are TensorFlow or PyTorch. If you want to go even further, you can try a technique called transfer learning, in which you finetune the last layers of a state-of-the-art pretrained deep CNN to solve the problem. This will most likely give you the best results, since you are leveraging the power of huge pretrained models with millions of parameters, which you could never train yourself.

Grading
At the end of the competition, you can either choose to provide us with a detailed (set of) notebook(s), or a full pdf report in which you explain what you did and why. If you choose to provide us with one or more notebooks, please make sure your code is very clean and well commented. Use the 'markdown' property of notebooks to explain your code in great detail. This means that we expect you to really dig deep into why you chose this or that approach, what are your findings for them and how they work. Try to be concise and direct in your explanations, but more is definitely better than less: we cannot read your mind on how well you understand and grasp the subject, so show us you understand machine learning on a practical and theoretical level. You will be first and foremost graded on your explanations and insights, not on your leaderboard results. Explain what techniques you started with, what worked, how you improved them etc. Try to mainly cover the successful parts of your work, but also briefly mention what you tried and discarded as unuseful as well. Include also a brief description of the more computational aspects of your work: did you use parallelization? A GPU? How do your algorithms compare computationally? We expect you to use lots of strong graphs and visualizations, as this is a core skill of a good data scientist and this helps you convey your results better. We expect at least separate sections on data exploration, preprocessing, model selection and tuning, your results and an analysis (don't forget the analysis with appropriate evaluation figures!).

If you want, you can also hand in an intermediary report/notebook, so that we can give some feedback before you hand in your final work. Feel free to do this before the end of 25 november, 23:59. This intermediary report/notebook will not be graded and is not mandatory, it is only presented as an option for extra feedback. If you submit an intermediate report, use the same structure as you would for your final report, and try to include some visualizations already. The better your intermediate work is, the more feedback we can provide you.

Upload your notebooks or pdf+code on Canvas. Your pdf/notebook name should at least contain your first name, your last name, your student number and an indication whether it's your final report or your intermediate one.

While getting a good score on the leaderboard is encouraged and most likely means you are doing well, remember that you will be graded based on the quality of your notebook explanations or your report, so explain your steps and choices clearly and discuss all the issues you have faced during this challenge. Don't wait on building your report until after the competition is done!

Deadline
The deadline for the competition and the report is Wednesday, January 15, 23:59.

General remarks and tips
Read the Kaggle pages (Description, Evaluation, Data, Rules), the example notebooks in the starting kit, the announcements etc., since they contain a lot of crucial relevant information. Reread them again when you are finishing the project to make sure you did not forget anything important.

Good luck!

Evaluation
Evaluation metric: Your submissions will be scored using the multi-class Log Loss (also called Cross-Entropy Loss) which is defined as:
$$L=-\frac{1}{N}\sum_{n=1}^N\sum_{c=1}^Cy_n^c\log(p_n^c)$$
where N is the number of training/test samples, C is the number of classes and p(n,c) is the probability that sample n belongs to class c as output of your classifier. y(n,c) is the true class probability that sample n belongs to class c. Note that y(n,c) = 1 if sample n belongs to class c and y(n,c) = 0 otherwise. ### Submission Format: Submission files should contain 13 columns: an 'Id' column identifying the id of the test sample, as well as 12 columns (named with snake case) denoting the class probabilities of the sample belonging to the corresponding class. All values should be separated by a comma and no space. There should be a row for every element of the test set. Make sure that your first row corresponds with the test sample `test_0001`, your second with `test_0002` etc. There is a submission helper function included in the starterskit folder which will create a valid submission for you. The submission csv file should look as follows:
<p>Id,chicken,elephant,fox,german_shepherd,golden_retriever,horse,jaguar,lion,owl,parrot,swan,tiger
1,0.0,0.1,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
2,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.3,0.5,0.0,0.0,0.0
3,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.7,0.0,0.2
...</p>

Dataset description
You are given two sets of data: one labeled training set, and one unlabeled test set. You have to train your models using the labeled training set (you are strongly advised to subdivide the training set with a good validation scheme), and then use your finetuned models to predict the labels of the test set. When you submit your predictions of the test set (according to the format specified in the 'Overview/Evaluation' section), you will immediately get your multi-class log loss score calculated on a portion of the total test set. If this score surpasses your previous submissions, it will be displayed on the public leaderboard, determining your ranking there. It is important to realize that your public score is only calculated on a small portion of the test set, so avoid overfitting on the leaderboard scores!

We, the hosts, can see the private scores based on the full test set, and at the end of the competition they will become public for everyone. It is not uncommon to see large shifts in the rankings when the private leaderboard scores become public, so you must definitely take this into account when building your model to avoid large rank drops.

You may assume that the distribution of the classes in the complete test set is similar to the distribution of classes in the train set. To get some bonus points on your report, I want you to briefly write down your answer to this question at the end of your report:

What would you change to your model if the distribution of the test set was uniform over the 12 classes? Would you use the same loss function, performance metric, train/validation/test splits etc. as you did before?

Files
train.zip contains the labeled training data. All animals are sorted under their respective labeled folder.

test.zip contains the unlabeled test data. Don't change the labels of these images, since the order in which you have to submit your solutions depends on the numbering here. Don't worry, in your notebook you can load everything in the right order and after your model predicts the outputs, you can use the submission helper function to create a valid submission.

starterskit folder containing

      - a notebook on the feature extraction and VBOW approach + making a submission

      - a tutorial notebook on classification using the logistic regression classifier from sklearn

      - a tutorial notebook on bias/variance and making a learning curve (disclaimer: use as is, it is copied over)

      - a notebook on the feature descriptors

      - a helper file and a file with functions creating the features

      - a folder called 'features', consisting of large pickle files which contain the precomputed feature descriptors for your convenience (created with settings equal to the ones in the 'create_VBOW' notebook)

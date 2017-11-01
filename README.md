# Data Scientist
Georgios Pligoropoulos

## Question 1

**Background**

One part of our work is to interact with customers and solve their problems. Keeping this in mind this exercise tries to simulate the following situation.

An automotive company contacted us with a problem related to an unusual vibration in the car while customers are driving. To minimize the impact in the end-user experience the company wants to identify the cars containing the problem before they leave the factory.

To solve the problem our client has provided several datasets (Q1 folder). In several meetings with some departments from our client several people hinted "From what I’ve seen I think the problem appears in the cars from March".

**The Task**

Please write some code (Python, R, MatLab) for the following questions.

1. Create a classification process to identify the problematic cars

2. Assess the classification process

Recommendations:

* Comments are more than welcome.

* Generate some outputs so we can rapidly verify the code (charts, variable outputs, etc.)

So we have:

* Cars.csv is when the cars were purchased and how much each one costs

* Car Sensor readings can be joined with the above and every row will be the car, purchase info and sensor info

* The "card id" column must be “car id” so I’ll treat it as a typo (assumption)

* Then the WarrantyDB.csv is our targets along with when they occurred

On the other hand if Card Id is NOT a typo then this means that our targets cannot be easily joined with the rest of the dataset.
So the easiest way is to solve this with the client and get the car ids that correspond to these warranty problem.
Now if we want to be hackerish about it there might be a relation between warranty date and purchase date, for example the subtraction might always give a constant value and thus easily identify them. This could be a domain knowledge that could easily be retrieved, perhaps any car owner in UK who is a little keen with cars knows about it.

Another thing that is missing from the story we are trying to built is when and how did the Car Sensor readings were read. It seems that where was a single read from all the sensors only once for every car. Was it when the warranty ended and had to be renewed?
These questions are vital to have a good sense of what we are doing and perhaps be able to do feature engineering.

Anyway let’s say that we overcome those issues and that we have all the features from the cars and cars sensors csv and we match them with binary targets, from warranty db.
More specifically the targets which are categories (classification) could be multi-class but since the car company only cares about the vibration issue we can consider only these cases and thus fallback to a binary classification problem.

Note that we immediately drop the features from the cars.csv because the date of purchase and how much the car costs might be unknown at time the company cares to check this vibration issue. From the description the company wants to know about the issue BEFORE the car leaves the factory. So we can consider that at that time the price of the car might be unknown and the purchase date might also be unknown. The cars could be stored in a warehouse before they are sold. This can be verified with the client.

We notice that we have more instances than the maximum car id and with further investigation we see some ids are duplicated. (we could check with the client if they have a unique identification). We check by code if these non-unique ids of cars.csv match with the ids of car sensor readings.
We find out that they indeed match so it seems that there is a row by row correspondence. We can re-index the dataset with our own serial number keeping a dictionary with the original keys.
This may pose a problem when we are trying to match with the targets. This is a communication issue that could be easily solved with the client as described above.

Note that we are being cautious of not throwing away instances since we have such a relatively small dataset.

We notice by eye that Sensor 14 is a constant value and thus provides zero information to our classification task. Of course we check by code for all features.

We basically have only numbers. Note as explained above that we have dropped the dates. As a first step we standardize our dataset to remove location and scale of each attribute (mean=0, std=1). This is also useful for PCA later.

Since we have a finite number of features it worths doing a distribution plot for each one of them to observe any peculiarities.

Next step is to plot again the distribution of a feature separately for the False and the True class within the same figure (color-coded) and see if some features provide useful information of our classification. This will easily reveal if the Naive Bayes classifier will be able to perform.

Doing pair-plots for all features could be useful but also time-consuming.

We use dimensionality reduction to either 2D or 3D by using PCA, MDS, IsoMap and t-SNE and see if any of them provides a good separation to the 2D or 3D space that we can observe.

We count that only 195 cases suffered from the vibration issue. 195/1385 ~= 14% of them are classified positively as having the vibration issue (given that we used this convention)

14% is small but not too small. If it was smaller we could see our classification problem from the perspective of an anomaly. In other words we could use an Isolation Forest to hopefully get as outliers the cars with the vibration issues. We are saying hopefully because this is not trainable. However if we get a good accuracy we could embed this in our model as a step before classification.

So we have an imbalanced classification problem and we do not have lots of samples. So one easy approach is to do upsampling or generating new data by data augmentation. Perhaps instead of using upsampling we could introduce some small gaussian noise to our inputs. The goal is to end up with a balanced classification problem to not bias any of the classifiers.

Then we have at our disposal several classifiers: Naive Bayes, QDA classifier, Random Forest, Support Vector Machine classifier, Logistic Regression, KNN classifier.
Note that we are not going to try an MLP neural network as classifier because of our limited dataset.

Before applying we are keeping ~20% of our dataset on the side for testing purposes. Note that whatever preprocessing we apply on our original dataset we keep it here as well.

We use cross validation to choose our model and the hyperparameters. We check if the variance of our folds of the k-folds of cross validation are relatively small. if it is we can safely take the mean, if not we are reshuffling the dataset and executing cross validation again multiple times if not computationally expensive. We choose either the model with the smallest CV score or the simplest model if its cv score is within one standard deviation from the minimum cv score.

We repeat the above process for the original dataset as well as the products of the dimensionality reductions. We try several dimensionalities if computational resources allow it.

We test the performance of our classifier against the testing set to see if it generalizes.
We have to start considering alternatives, such as building ensemble models with boosting, if the classification accuracy is worse than we could tolerate.

If all ok we retrain our best model on the entire dataset (train + testing) in order to retrain it given a larger dataset.

Now the model is ready to classify new cars that come out of production line and before they leave the factory.

## Question 2

We believe that designing truly innovative machine learning algorithms is as much a creative pursuit as a technical one. This challenge is designed to see how you approach a non-trivial problem, and whether you can come up with an interesting and creative feature set.

 

**The problem:**

To prevent **_attachments_** being emailed to the incorrect recipients.

 

**The data set:**

A JSON array of emails in the following format:

 

<table>
  <tr>
    <td>{
    sender: "biil@cr.com",
    recipients: ["bob@cr.com", "jane@cr.com"],
    subject: "Project zebra meeting",
    timestamp: "2016-01-01 13:48:32",
    attachments: [
        {name: "project zebra board meeting minutes.pdf",
         content: "Blahblah hjghjbhbblah"},
        {name: "project zebra offer letter.pdf",
         content: "Blah blah blah"}
    ]
    body: "All – please see notes ahead of meeting on Thursday.
          There’s a document we need to approve to set up there.
          Regards,
          B"
}</td>
  </tr>
</table>


 

**The task:** 

1. Write a small document explaining your approach to flag emails containing the attached files to the wrong recipients.

2. Given everything can be always improved. List features/experiments (maximum 10) you would like to explore further if you were to be given more time to work on this problem. In this case, you would test the hypothesis that these features could be used to improve the process described in the previous question. For each feature/experiment write a short description of why/how it would improve the process.

 

Basically we have a binary classification problem and we would like to assign True if attachments are suitable for the recipient and False if attachments are NOT suitable for the recipient.

If we have reliable targets, and this is a big if, we would NOT evaluate simply by checking how high is our classification accuracy or F1 score because now we want to treat the False of much higher importance. Having an attachment reaching to the wrong hands could be detrimental.

So in order to be careful we will follow this strategy: Even if one attachment does not belong to the recipient then we will treat this email as incorrect. From all attachments we will consider the worst case and assign accordingly. This means that the above json will not be converted to a single instance. Since the example has two attachments it will be two instances and the body, subject, recipients etc. will be repeated for these two instances.

We treat the same way the case of multiple recipients. So above we have two recipients which makes generates two instances.

So overall with two recipients and two attachments we would have 2x2 = 4 instances for this email. This could cause memory issues since we could generate many instances so probably we would need to build a dataprovider that does not require all of the data to be represented in a table.

However this upsampling of the data could mean that one email is represented much more on the whole of our dataset than others, especially in extreme cases of many attachments or many recipients.
Our main problem is to prevent having the wrong attachment for a single recipient so on that perspective we could say that our approach so far makes sense.

Let’s say that we indeed want to approach it as a supervised classification problem and we do not have targets then we have several options according to the UX we want to offer. We could gather targets explicitly or implicitly.

Explicitly works by asking users to provide the target (click on a button) to inform us whether the attachment was the correct or not.

Implicitly could be by checking the next email within the same day for another recipient but with the same attachment. If this next email included an "excuse me" or “sorry” or “attaching again” etc. we could classify the original email as incorrect (False).
In fact we could use a perhaps already existing model such as sentiment analysis to get if the sentiment of the email is that of an apology.

So now we consider that we have enough targets.

Since this is text, sentiment analysis output could be useful feature(s).

Another typical feature is to use Tf-Idf for the words by filtering out first the most common ones that occur in almost all emails like "and", “to”, etc.

Levenshtein distance gives the distance between two words. So for comparing two sentences we would have all words of one sentence with all words of other sentence. So for comparing a sentence with 5 words to a sentence with 6 words we could first filter them to remove common words and as a result get for example 3 and 4 words and then  the combinations would be 4x3.

This is one metric that I can think of right now and there are more, but the main idea is to compare the subject of the email with the filename of all the attachment. Even if it is not directly related to our main problem it is in general useful to denote the inconsistency between email and attachment.

Another thing that we could try is that we could consider that certain people send certain kinds of emails to other people. For example the employer sends a licence of the newly bought software to the employee but never happens in reverse. This would consider the email subject and email body.

So first of all we could use K-means or hierarchical clustering in an unsupervised way to cluster kind of emails together in email categories. Some manually filtering beforehand could be useful.

Then we could create tuples with (sender, recipient, email category)

This would give us an extra feature we could add for every email instance that includes this sender and this recipient.

And of course there are more things and ideas to explore. For example from my own experience the body of the attachment could be anything and therefore not be useful but depending on the dataset you have in hand it could be meaningful to be taken into account. For example certain colleagues could only mainly be sending attachments of source code and therefore a PDF of a literature book could be an anomaly to that.

In fact I would create as many of these ideas as possible and test the feature individual to see if they are strong enough and with the rest of the dataset together and the result would be a list with false positives. How many times we classified something as True (the attachment is ok) but in reality it was False and we let a wrong attachment go through. ← we are trying to minimize this false positives.

So seeing what works better from this pet examples, with relatively small number of dataset (emails could easily be millions) we would build intuition on how to work futher.


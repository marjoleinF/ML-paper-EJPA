---
title: "Machine learning and prediction in psychological assessment: Some promises and pittfalls."
bibliography: paper.bib
csl: apa.csl
output:
  word_document:
    reference_docx: reference.docx
    keep_md: TRUE
---





"When we raise money it’s AI, when we hire it's machine learning, and when we do the work it's logistic regression."

(Tweet by bio-statistician Daniella Witten; original author unknown)



<!-- To cite, use @key. To put citations in parentheses, use [@key]. To cite multiple entries, separate the keys by semicolons, e.g., [@key-1; @key-2; @key-3]. To suppress the mention of the author, add a minus sign before @, e.g., [-@R-base] -->

# Abstract

Modern prediction methods from machine learning (ML) and artificial intelligence (AI) are becoming increasingly popular, also in the field of psychological assessment. These methods provide unprecedented flexibility for modeling large numbers of predictor variables, and non-linear associations between predictors and response. In this paper, we aim to take a look at what these methods may contribute for the assessment of criterion validity, and what their possible drawbacks are. We apply a range of modern statistical prediction methods to a dataset for predicting the university major completed, based on the subscales and items of a scale for vocational preferences. The results indicate that logistic regression combined with regularization performs strikingly well already in terms of predictive accuracy. More sophisticated techniques for incorporating non-linearities can further contribute predictive accuracy and validity, but often marginally.    



# Introduction 

Machine learning (ML) and artificial intelligence (AI) are by now familiar buzzwords in many fields of empirical research, including psychology. In the field of psychological assessment, interest and application of these methods is also increasing. We believe ML and AI have the potential to contribute to our field, but the buzz around these methods can be reminiscent of the tale of the emperor’s new clothes. We believe when it comes to ML and AI, the emperor is in fact wearing clothes, but they are often not so new. Many of the techniques presented as machine learning (e.g., cross validation, regularization, ensembling) have long been known and fruitfully applied in statistics, psychometrics and psychological assessment. 

In the current paper, we look at how several modern methods from statistics, ML and AI may contribute to our field, and what their limitations are. Note, we will use the term statistical learning to refer to both traditional and more recent (sometimes referred to as ML or AI) tools for data analysis. As already suggested by the motto at the start of this paper, there is no consensus on whether specific methods are statistical, AI or ML, so we avoid making the distinction altogether. We focus instead on the aim shared by all these methods: Learning from data. We focus on methods for prediction of a response (dependent, criterion) variable, often referred to as supervised learning methods. Thus, unsupervised learning methods (e.g., factor analysis, clustering, correlation networks, topic models from natural language processing) are outside the scope of the current paper.





## Recent shifts in statistical learning

Modern developments in statistical learning methodology have yielded two main shifts:   

1. Increased focus on prediction.

2. Increased flexibility: Modern methods allow for capturing non-linear associations and/or modeling large numbers of predictors.

We believe that the first shift is highly beneficial for our field, because prediction of behavior is one of the core tasks of psychological assessment. Accurate evaluation of predictive accuracy is needed to provide evidence for the validity of test score interpretations, but also when more complex decision systems are developed for data-driven decision making. Traditionally, the field of psychology at large has been mostly interested in *explanation*, or developing and testing theories of human behavior. This has sometimes led researchers to overlook *prediction*, perhaps because their main aim was to explain behavior. A theory, however, can only explain real-world phenomena to the extent that it can accurately predict them [@YarkyWest17]. 

The traditional focus on explanation may have motivated researchers to compute effect sizes (e.g., *R^2^*, Cohen's *d*) on *training* data; that is, using observations that were also used to fit the model. This leads to overly optimistic effect size estimates. More realistic effect sizes can be obtained, for example, through cross validation: By computing effect sizes on a sample of observations *not* used for fitting the model [@RooiyWeed20]. It is interesting to note that cross validation has been discussed in the field of assessment for almost a century [@Lars31; @Mosi51], but its use has become more common only in recent years. 

We believe that the second shift, towards flexibility, brings both promises and pitfalls for our field. Promises, because few if any real-world phenomena behave in a purely linear and additive fashion. Pitfalls, because assumptions of linearity and additivity (i.e., no interactions) are very powerful when it comes to inference and interpretation, even if they are known to be only partialy true. This means that the often one-sided focus on maximizing predictive accuracy in AI and ML are of limited value when it comes to understanding and explaining behavior, and the role of these methods is, at best, in hypotheses generation. 

Of note, unrestricted flexibility leads to overfitting and poorly generalizable results. In statistical learning, this has been formalized in the *bias-variance trade-off*. Informally, this trade-off states that the more flexible a model is allowed to approximate any possible shape of association between predictors and response (i.e., the lower the *bias*), the worse the model will generalize to new samples from the same population (i.e., the higher the *variance*). To obtain optimally generalizable results for a given data problem and sample (size), bias and variance should thus be carefully balanced through choosing an appropriate model-fitting procedure.  

Bias can be increased and variance reduced in various ways, including: 

* Limiting the complexity of the functional form (e.g., model only linear associations; model only main effects);

* Limiting the number of potential predictors used (e.g., include only few predictors; use sum or factor scores instead of item scores as predictors);

* Regularized estimation procedures (e.g, lasso, ridge, or elastic net regression; use of Bayesian priors); 

* Ensembling (e.g., in psychometrics, multiple items are often aggregatec into subscale or factor scores; in ML predictions of so-called base learners are often aggregated into the predictions of an ensemble). 

If the bias is well-chosen and realistic, generalizability of the fitted model will be improved. In other words: we can buy predictive power by making realistic assumptions. If the bias is not well chosen, predictive accuracy and generalizability will obviously suffer. 




# Empirical example

We aim to illustrate and compare the use of a range of statistical learning techniques through a data-analytic example. We focus on a predictive validity question: To what extent do the item and subscale scores on a measure of vocational preferences predict the type of university major completed? Note, we will not focus on substantive aspects of this prediction problem, but we use it to illustrate more general principles of flexibility, overfitting and interpretability in predictive modeling in assessment. In test development, providing evidence for criterion validity of the scores is vital as it often is used by practitioners to choose between existing tests. Thus, establishing test-criterion related evidence is a fundamental part of test construction. Therefore, it seems obvious that the potential of statistical learning procedures should come to bear here. 

Readers interested in replicating our analyses will find our annotated code and results in the ESM.




## Method

### Dataset



We use a dataset from the Open Psychometrics Project (https://openpsychometrics.org/_rawdata/). Data were collected through their website from 2015 to 2018. Respondents answered items on vocational preferences, personality and sociodemographic characteristics. The sample likely does not represent a random sample from a well-defined population, which would normally be required for evaluating a test's validity.  

We investigate predictive validity of the RIASEC vocational preferences scales [@LiaoyArms08]. The RIASEC uses six occupational categories from Holland's Occupational Themes [@Holl59] theory: Realistic (R), Investigative (I), Artistic (A), Social (S), Enterprising (E), and Conventional (C). There are 8 items for each category, each describing a task (e.g., R6: "Fix a broken faucet" or I2: "Study animal behavior"), to which respondents answer on a 1-5 scale, with 1=Dislike, 3=Neutral, 5=Enjoy. The items are presented in Appendix A. The research question from an assessment perspective is whether the RIASEC scores can be used to predict the university major completed. Such evidence could support the use of the scale in applied settings; moreover, the results could inform decision rules.

From the full dataset, we selected participants who completed at least a university degree, yielding a sample of *N =* 55,593 observations. As the criterion we take a binary variable, indicating whether respondents majored in Psychology (19.42\%), or in a different topic (80.58\%). Further descriptive statistics of the sample are presented in Appendix B. 








### Model fitting and evaluation

We fitted a range of traditional and more recent (ML/AI) methods to model the relation between the RIASEC scores and the criterion. This will show the magnitude of differences in performance such algorithms typically yield. Also, it exemplifies the researcher degrees of freedom in such cases and it is thus important to use separate data for fitting and evaluation of the models.  

We separated the data into 75\% training observations and 25\% test observations. Our training sample thus consists of 41,694 respondents, of which 19.46\% majored in psychology. Our test sample consisted of 13,899 respondents, of which 19.3\% majored in psychology. Other train and test sample sizes may sometimes be preferred, or $k$-fold CV. Considering the current sample size, however, we do not expect the results to be very sensitive to this choice. 

All analyses were performed in **`R`** [version 4.1.0, @R21]. We tuned the model-fitting parameters for all models using resampling and cross validation (CV) on the training observations. We did *not* tune the parameters of the generalized additive models (GAMs), because we expected the defaults to work well out of the box. The specific packages used, as well as the code and results of tuning and fitting the models are provided in the ESM. 

We evaluated predictive accuracy of the fitted models by computing the Brier score on test observations. The use of accuracy measures derived from the confusion matrix of actual and predicted classes, like the misclassification error, sensitivity (or recall), positive predictive value (or precision) are pervasive in the machine learning literature. However, these measures disregard the quality of predicted probabilities from a fitted model and we therefore recommend against their use for evaluating predictive accuracy. Methods for predicting a binary outcome should not only provide a predicted class, but also a predicted probability to quantify the uncertainty of the classification. To evaluate performance, the quality of this probability forecast should thus be evaluated [@GneiyRaft07]. 

The Brier score is the mean squared error of the predicted probabilities: 

$$\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{p}_i)^2$$

Where $y_i$ is the observed outcome for observation $i$, taking a value of 0 or 1; $\hat{p}_i$ is the model's predicted probability. We computed Brier scores on training as well as on test observations; thus $N$ can be taken to be the training or the test sample size. A Brier score equal to the variance of $y$ indicates performance no better than chance (in the current dataset, the variance was 0.1946 for training and 0.193 for test data). To obtain a pseudo-$R^2$ measure, we take 1 minus the Brier score divided by the variance of $y$, which takes values between 0 (indicating performance no better than chance) and 1 (indicating perfect accuracy). 




## Results



Considering the two shifts in predictive modeling discussed in the Introduction, we fitted all models twice: Once using subscale scores, once using item scores. This allows us to evaluate whether our conclusions generalize between the two approaches, and to gauge the effect of having a larger pool of predictor variables (which are likely more noisy but possibly more informative of the criterion).


### (penalized) Logistic regression

Our benchmark traditional method is an additive generalized linear model (GLM): Logistic regression. If CV results indicated predictive accuracy could be improved by application of a lasso or ridge penalty, we applied it. For prediction with subscale scores, no penalization was found to be optimal. The estimated coefficients for the subscale scores are presented in Figure 3; as expected with the currently large sample size, all subscale scores obtained $p$-values $< .001$. The strongest effect was a positive effect from the Social preferences scale, and the weakest effect was a negative effect from the Conventional preferences scale.

For prediction with item scores, CV indicated optimal performance for a small but non-zero value of the lasso penalty. With an increasing number of predictor variables, this beneficial effect of penalization (or regularization) is generally expected. The resulting item-level coefficients are depicted in Figure C1 (ESM). The item coefficients indicate similar relevance of the subscales as the previous analysis, but provide a more finegrained view of individual item's contributions.









### Generalized additive model

Next we fitted generalized additive models (GAMs) with smoothing splines. Smoothing splines allow for flexibly approximating non-linear shapes of association between predictor and response. At the same time, overfitting is prevented by penalizing the wigglyness of the fitted curves. The splines provide a flexible but smooth approximation to the observed datapoints, while the additive structure provides ease of interpretability because the estimated effects are *conditional* (i.e., keeping the values of all remaining predictors fixed). @BrinyHama17 provide a more detailed introduction to GAMs aimed at psychologists.





**Figure 1**  
*Fitted smoothing spline curves for each of the RIASEC subscales*
![](Paper_files/figure-docx/unnamed-chunk-8-1.png)<!-- -->
*Note. * Values on the $y$-axis reflect the effect on the log-odds of having completed a university major in psychology.  
  
  
The splines fitted to the subscale scores are presented in Figure 1. Similar to the GLM, we see positive effects of the Social and Investigative subscales, and negative effects of the Realistic, Artistic, Enterprising and Conventional subscales. The Social preferences subscale shows a near-linear effect, while the other subscales' effects clearly exhibit some stronger non-linearity. An advantage of GAMs is that they allow for inference: they provide $\chi^2$ tests to evaluate the significance of the effect of each predictor variable. As expected with the current large sample size, all subscale scores obtained $p$-values $< .001$.

For the GAM fitted using item scores, we also applied penalization, as this was expected to be beneficial for prediction, as similarly observed in the GLM. We do not depict the fitted curves for space considerations here, but Figures 3 and C1 (ESM) show the $\chi^2$ values per subscale and per item, respectively. The figures indicate very similar effects of the predictors, between the (penalized) GLMs and GAMs.





We now leave the realm of additive models, and set about fitting models that allow for capturing interaction effects:



### Decision tree

We fit a single decision tree using the conditional inference tree algorithm [@HothyHorn06]. This algorithm eliminates the variable selection bias present in many other decision-tree algorithms. Decision-tree methods and variable selection bias are discussed in more detail by @StroyMall09, who provide a comprehensive introduction aimed at psychologists. According to our CV results with the subscales as predictors, a tree depth of seven was optimal, yielding a tree with $2^6 = 128$ terminal nodes. Thus, for this data problem, the most accurate tree is surely not the most interpretable. The predictor variables selected by the trees are depicted in Figures 3 and C1 (ESM).

For illustration, Figure 2 shows the decision tree fitted to the subscale scores, pruned to a depth of three. The Social, Realistic and Enterprising preferences subscales were used in the first splits of the tree. The bars in the terminal nodes depict the proportion of participants within each node that majored in psychology. Thus, the Social subscale shows a positive effect, and the Realistic and Enterprising subscales show a negative effect. With regards to possible interactions, note that split number 10 suggests that the Enterprising subscale appears relevant only for higher values of the Social, and lower values of the Realistic subscales. However, such a split may also reflect additive effects combined with multicollinearity. Although decision trees can *capture* interaction effects, they cannot be straightforwardly used to statistically test their significance; a disadvantage shared by virtually all flexible ML and AI techniques.  
  
  
**Figure 2**  
*Conditional inference tree pruned to a depth of three*
![](Paper_files/figure-docx/unnamed-chunk-11-1.png)<!-- -->





Although decision trees are easy to interpret, they suffer more strongly from instability than GLMs and GAMs. With instability, we mean that a small change in the training data can lead to large changes in the resulting model. The cause of this instability partly lies in the rather rough cuts made in the tree. Tree ensembling methods capitalize on this instability. They derive a large number of learners (e.g., trees), each fitted on different versions of the training dataset. Different versions of the training data can be generated, for example, by taking bootstrap samples from the training data, a method also known as bagging. More powerful tree ensembling methods are random forests and boosting. Introductions about tree ensemble methods aimed at psychologists can be found in @StroyMall09 and @MillyLubk16.


### Gradient boosted tree ensemble

The first tree ensemble method we apply to the data is a gradient boosted ensemble. Boosting uses sequential fitting of so-called weak learners to create a strong learner. Weak learners are simple models, that provide predictive accuracy (slightly) better than chance. When boosting trees, we use weak learners in the form of small trees, with only a few splits. Sequential learning means that each consecutive tree is adjusted for the predictions of previous trees. In effect, observations that were well (badly) predicted by previous trees receive less (more) weight when fitting the next tree.












A disadvantage of decision tree ensembles is their black box nature: While individual trees are generally easy to interpret, an ensemble of trees is impossible for humans to grasp. Therefore, so-called variable importance measures have been developed for interpretation of tree ensembles, which aim to quantify the effect of predictor variable on the predictions of the ensemble. In this paper, we use the permutation importances proposed by Breiman [-@Brei01]. These quantify how much an ensemble's predictive accuracy would be reduced, if the values of each of the predictor variables are randomly shuffled. The variable importances of the fitted gradient boosting ensembles are depicted in Figures 3 and C1 (ESM).

Importance measures provide a useful ranking of the contributions of each predictor to the ensemble's predictions, but should be interpreted with care. They should not be used to judge the significance of the effect of predictors; tree ensembles can easily include predictors in the model which in fact have no effect on the outcome. Furthermore, there are many ways to compute variable importance measures, which each may yield different conclusions, especially when predictors are correlated [@StroyBoul07; @StroyBoul08; @Nico11; @NicoyMall10]. Especially with correlated predictors, permuting the values of predictor variables may lead to unrealistic data patterns. These issues illustrate the interpretability problems which come along with complex prediction methods such as tree ensembles, support vector machines and (deep) neural networks.  



### Random forest

Another popular decision-tree ensembling method are random forests [@Brei01]. Like boosted tree ensembles, random forests fit a large number of decision trees. The ensemble's predictions are simply the average over the predictions of the individual trees. Random forests do not employ sequential learning: each tree is fitted without adjusting for predictions of the other trees in the ensemble. Unlike boosting, random forests employ trees with many splits: in the original algorithm of [@Brei01], trees were grown as large as possible. Later studies, however, have shown that large trees can lead to unstable results when there are many correlated predictors that are at best weakly correlated to the response [@Sega04]. It is thus beneficial to grow large, but not too large trees. 

The most characteristic feature of random forests is how it selects variables for splitting: A random sample of *mtry* candidate predictor variables is considered for every split in every tree. From this set of predictor variables, the best splitting variable and value is selected. Without random selection of variables, each tree of the ensemble would likely use the same set of relatively strong predictors, and thus be very similar. Averaging over many very similar trees is unlikely to improve predictive accuracy. Thus, the randomization makes the trees more dissimilar, which likely improves performance of the ensemble. 

The variable importances of the fitted random forests are depicted in Figures 3 and C1 (ESM).












### Prediction rule ensembling

Prediction rule ensembles (PRE) aim to strike a balance between the high predictive accuracy of decision tree ensembles, and the ease of interpretability of single decision trees and GLMs [@Fokk20; @FokkyStro20]. The method fits a boosted decision tree ensemble to the training dataset, and takes every node from every tree as a rule. For example, membership of Node 2 in the tree in Figure 2 can be coded using a single condition: *Social $\leq$ 27*. Membership of Node 14 involves multiple conditions: *Social > 27 & Realistic > 17 & Realistic $\leq$ 24*. Each of these nodes can be seen as a dummy-coded rule, which takes a value of 1 if the conditions apply, and 0 if not.

PRE applies lasso regression on a dataset consisting of both these rules and the original predictor variables. As such, it combines the strengths of penalized regression and tree ensembles. Although the boosted decision tree ensemble will initially contribute a large number of nodes (rules), use of lasso regression will give many of these rules a weight of zero, which removes them from the final ensemble. As such, PRE provides a sparse and interpretable final model.  

The PRE we fitted using the subscale scores consisted of 48 rules, providing a great simplification compared to the $>500$ trees of the boosted ensemble and random forest. Note that the current dataset is exceptionally large, which tends to result in longer rule lists when only predictive accuracy is optimized, because very large samples allow for capturing highly nuanced effects. In Table 1, the six most important rules are shown.
 


**Table 1**  
*Six most important rules in the prediction rule ensemble*

|Description                         | Coefficient |
|:-----------------------------------|:-----------:|
|Soci > 27 & Ente <= 31 & Conv <= 30 |    0.182    |
|Soci > 23 & Ente <= 29 & Real <= 24 |    0.181    |
|Real > 10 & Soci <= 35              |   -0.175    |
|Real <= 22 & Soci > 19 & Inve > 18  |    0.138    |
|Inve > 10 & Real <= 13              |    0.120    |
|Conv <= 23 & Arti <= 29 & Soci > 21 |    0.112    |

Note that each rule has obtained an estimated coefficient, which are simply logistic regression coefficients: They reflect the expected increase in log-odds if the conditions of the rule apply. PRE also provides variable importance measures, which are presented for the fitted ensembles in Figures 3 and C1 (ESM). An introduction to PRE aimed at psychologists is provided in @FokkyStro20. 







 


### *k* Nearest neighbours

A prime example of a highly flexible method, perhaps the most non-parametric method of all, is the method of *k*-nearest neighbours (kNN). In fact, kNN does not even fit a model; it merely remembers the training observations. To compute predictions for new observations, kNN computes the distance of a new observation to all training observations, in order to find the *k* nearest ones (the neighbours). It then takes the mean of the response variable over these $k$ neighbours as the predicted value. This provides the greatest possible flexibility of all prediction methods, as it does not impose *any* a-priori restriction on the shape of association between predictors and response. This flexibility is both the strength and weakness of kNN: with increasing numbers of of predictor variables, the performance of kNN worsens fast. Only in lower dimensions is the great flexibility of kNN beneficial. 

kNN has only a single tuning parameter: *k*. With larger values of *k*, the predicted value for a new observation averages over a larger number of observations (neighbours). Thus, higher values of $k$ yield lower variance, but higher bias. Furthermore, because kNN is a fully distance-based method, in which all variables obtain the same weight of 1, the method does not provide *any* measure of effect of individual variables, and we thus do not plot variable contributions for kNN here.












## Model comparisons

### Variable contributions

Figure 3 depicts the variable contributions in the models fitted using RIASEC subscale scores. Note that the coefficients of the logistic regression reflect both direction and strength of the effects. For the other models, the variable contributions only reflect the strength of the variables' effects. Figure 3 shows similar variable contributions for all methods: The Social preferences are most important for predicting university major completed, followed by Realistic, followed by Enterprising preferences, while the Conventional and Artistic subscales contribute least. The variable contributions for models fitted using the item scores as predictors yielded similar conclusions and are provided and discussed in Figure C1 (ESM). 




  
  
**Figure 3**  
*Variable contributions for each of the models fitted using RIASEC subscale scores as predictors*
![](Paper_files/figure-docx/unnamed-chunk-29-1.png)<!-- -->![](Paper_files/figure-docx/unnamed-chunk-29-2.png)<!-- -->
*Note.* Coefficients in the logistic regression and importance measures of the prediction rule ensemble are on the scale of standard deviations. Importance measures for the other methods are on the scale of variances; for those methods, the square roots are plotted.  
  
  

  
  
**Figure 4**  
*Predictive accuracy on train and test observations for each of the models fitted on subscale scores (left panel) and items scores (right panel)*
![](Paper_files/figure-docx/unnamed-chunk-30-1.png)<!-- -->
*Note.* (p)GLM = (penalized) logistic regression; GAM = generalized additive model with smoothing splines; PRE = prediction rule ensemble; GBE = gradient boosted tree ensemble; RF = random forest; kNN = k nearest neighbours.  
  
In Figure 4, pseudo-$R^2$ values on train and test data are depicted with confidence intervals. Note that the confidence intervals for test data are systematically wider than for training data, but this is mostly due to the much larger number of training observations.

The left panel of Figure 4 shows that with the subscale score, the best test set performance was obtained with the boosted tree ensemble, and very closely followed by the generalized additive model, prediction rule ensemble, random forest, $k$ nearest neighbours, logistic regression, and finally the decision tree. This latter result is rather unsurprising: a single decision tree is generally expected to have somewhat lower predictive accuracy, but they often 'win' in terms of interpretability, which can be observed in Figure 4, which shows that the decision tree uses only about half of the items for prediction. The boosted tree ensemble performing best is also not very surprising, giving its top-ranking performance in forecasting competitions. 

From the left panel in Figure 4, we obtain the following take-aways:

1. On the test data, none of the methods performs significantly worse or better than any of the other methods.

2. The difference between training and test performance increases with increasing flexibility. The methods that incorporate linear main effects (logistic regression, GAM, PRE) show the smallest difference in performance between training and test data. These methods thus appear least likely to overfit.

3. The more flexible methods (single tree, kNN, boosted ensemble, random forest) show greater susceptibility to overfitting.

The subscale scores did not provide strong predictive power, with $R^2$ indicative of a moderate effect. Using item scores as predictors yielded a substantial (about 50\%) increase in variance explained. Again, best performance on the test data was obtained with the boosted tree ensemble. This time, it was followed by the prediction rule ensemble, then the generalized additive model, logistic regression, random forest, $k$ nearest neighbours, and finally the decision tree.

From the right panel in Figure 4, we can add to our earlier take-aways:

4. With a larger number of predictors, differences in performance between the methods become more pronounced, but none of the more sophisticated methods significantly (or substantially) outperforms the GLM with lasso penalty.

5. With a larger number of predictors, the difference in performance between training and test data becomes more pronounced. Higher dimensionality creates more opportunity for overfitting, even though all methods feature powerful built-in overfitting control.




# Discussion

Our conclusions can be succinctly summarized as: Logistic regression is hard to beat. Linear main effects models (i.e., (penalized) GLMs) tend to capture most of the explainable variance. This finding corresponds to a range of previous studies noting a lack of (substantial or significant) benefit of sophisticated machine learning methods over (penalized) regression, in prediction problems from psychology and medicine [e.g., @ElleyMcDo20; @LittyCook21; ChriyJie19; @GraveyNieb20; @NusoyTham20; @LynayDenn20]. 

Sophisticated methods can only improve upon linear main-effects models by capturing more nuanced non-linearities and interactions. Almost by definition, these effects are of smaller size. Capturing these smaller, more nuanced effects comes at the price of an increased tendency to overfit. To reliably approximate small effects, much larger sample sizes are needed. Even if sophisticated methods outperform simpler methods like logistic regression in terms of predictive accuracy on test data, their tendency to overfit and their black-box nature may make them less suited for increasing scientific understanding, and/or making influential decisions about individuals (e.g., clinical or selection settings).

Perhaps GAMs and PREs may provide the most steady improvement on (penalized) GLMs. They are essentially GLMs with added flexibility for capturing non-linearities, but provide robust overfitting control and also retain interpretability. Especially GAMs may provide the 'best of both worlds': They provide the flexibility of modern statistical learning, robust overfitting control and allow for performing statistical inference. Most flexible machine-learning methods especially fall short in terms of the latter, which limits their use for increasing scientific understanding and theory development. 

Our finding that item scores can provide better predictive accuracy than subscale scores corresponds to previous studies [e.g., @SeebyMott18; @StewyMott21]. As also noted by @Yark20, a large number of item scores will outperform any predictive model fitted on subscale scores, given a large enough sample size. At the same time, a handful of subscale scores is easier to interpret and use than hundreds of personality items. Also, with smaller samples (e.g., $N =$ 300 or 500), including prior knowledge about the subscale structure, through the use of subscale or factor scores, may likely improve predictive accuracy [@RooiyKarcUR].

Big-data applications involving, for example, image-, video- and text-based analytics may exhibit stronger patterns of non-linearity and interaction than the analytic example presented here. More sophisticated methods like deep neural networks may even be called for in such applications. However, similar rules of sampling and statistics apply in such applications: The more nuanced the patterns that we want to capture, the larger the sample sizes required. Sample size requirements for artificial neural networks by far exceed the sample sizes common in our field [e.g., @AlwoyCran18]. There is no doubt that image, text, audio, video and sensor-based data (will) provide novel ways of assessing psychological traits [@GillyRutl21; @BoydyPasc20]. Their relatively unobtrusiveness opens up new avenues for assessment, but the black-box nature of algorithms that can capture complex non-linear effects also brings ethical risks [@Rudi19; @BoydyPasc20]. 

The focus on predictive accuracy brought about by recent statistical, ML and AI methods is beneficial for the field of assessment. We should, however, guard against a blind focus on maximizing predictive accuracy on test observations, as this disregards two important issues:

* Data points analysed in, for example, research settings or forecasting competitions may likely differ from the data points that the predictive model will be applied to in practice. These differences may be subtle in relatively closed, low-stakes systems, like online recommender systems. Much psychological assessment is, however, focused on offline, out-of-lab human behavior, often with high stakes. Generalizing research findings to the real world remains difficult; external validity has not become irrelevant all of a sudden. Gains in predictive accuracy in controlled research settings may be swamped by practical aspects of data problems, like population drift, measurement error, ethics, interpretability, and data-collection costs [@Hand06; @Efro20; @LuijyGroe19; @Raut20; @FokkySmit15].  

* From both an ethical and scientific perspective, validity has become more (not less!) important with newer and bigger data sources. A blind focus on predictive validity leads to black-box assessment procedures with limited content, internal and construct validity. For opening the black box, there is an important role for the field of psychological assessment and psychometrics. Not only by applying our existing theory, evidence and methods, but also by continually improving, adopting and developing them [@BleiyHopw19; @TayyWoo20; @AlexyMulf20; @IlieyGrei19].  

Finally, although modern statistical prediction methods have certainly improved our ability to predict, attribution and interpretation have not become easier. Attribution (assigning significance to individual predictors) requires strong individual predictors and large sample sizes [@Efro20]. This task only becomes more difficult when datasets contain increasing numbers of predictors with modest effects. The task also becomes more difficult with methods that can capture increasingly nuanced non-linear and interaction effects. A range of interpretation tools for black box models have been proposed (e.g., variable importances, LIME, Shapley values, SHAP). However, the accuracy of their explanations cannot be quantified [@RossyHugh17; @CarvyPere19], and their inner workings pose another black box to most users, resulting in misinterpretation and misuse [@Rudi19; @KauryNori20; @WaayNieu21; @KumayVenk20]. With large numbers of predictors, fitted models become inherently difficult to interpret and black-box interpretation tools are unlikely to help with this. Thus, while flexible models might help to inform theory building, their use for making decisions in assessment procedures aimed at individuals is currently limited. 














  

\newpage
# References

<div id="refs"></div>


\newpage
# Appendix A: RIASEC items


| **The following items were rated on a 1-5 scale of how much they would like to perform that task, with the labels 1=Dislike, 3=Neutral, 5=Enjoy:**|
| ------------------------------------- |
| R1	Test the quality of parts before shipment |
| R2	Lay brick or tile |
| R3	Work on an offshore oil-drilling rig |
| R4	Assemble electronic parts |
| R5	Operate a grinding machine in a factory |
| R6	Fix a broken faucet |
| R7	Assemble products in a factory |
| R8	Install flooring in houses |
| I1	Study the structure of the human body |
| I2	Study animal behavior |
| I3	Do research on plants or animals |
| I4	Develop a new medical treatment or procedure |
| I5	Conduct biological research |
| I6	Study whales and other types of marine life |
| I7	Work in a biology lab |
| I8	Make a map of the bottom of an ocean |
| A1	Conduct a musical choir |
| A2	Direct a play |
| A3	Design artwork for magazines |
| A4	Write a song |
| A5	Write books or plays |
| A6	Play a musical instrument |
| A7	Perform stunts for a movie or television show |
| A8	Design sets for plays |
| S1	Give career guidance to people |
| S2	Do volunteer work at a non-profit organization |
| S3	Help people who have problems with drugs or alcohol |
| S4	Teach an individual an exercise routine |
| S5	Help people with family-related problems |
| S6	Supervise the activities of children at a camp |
| S7	Teach children how to read |
| S8	Help elderly people with their daily activities |
| E1	Sell restaurant franchises to individuals |
| E2	Sell merchandise at a department store |
| E3	Manage the operations of a hotel |
| E4	Operate a beauty salon or barber shop |
| E5	Manage a department within a large company |
| E6	Manage a clothing store |
| E7	Sell houses |
| E8	Run a toy store |
| C1	Generate the monthly payroll checks for an office |
| C2	Inventory supplies using a hand-held computer | 
| C3	Use a computer program to generate customer bills |
| C4	Maintain employee records |
| C5	Compute and record statistical and other numerical data |
| C6	Operate a calculator |
| C7	Handle customers' bank transactions |
| C8	Keep shipping and receiving records |


\newpage
# Appendix B: Uni- and bivariate sample descriptives



The total sample consisted of 55,593 participants. Mean age was 33.06 (SD = 11.68). With respect to gender, 67% of participants reported female, 32% reported male, 1% reported "other". With respect to marital status, 58% of participants reported being never married, 33% reported being currently married, 8% reported being previously married. With respect to type of area lived in when a child, 22% of participants reported rural, 37% reported suburban, 40% reported urban. With respect to language, 66% of participants reported that English is their native language, 34% reported that English is not their native language. With respect to race, 20% reported Asian, 1% reported Arab, 7% reported Black, 61% reported White, Native American, or Indigenous Australian (note that these three options were merged due to a coding mistake), and 1% did not report their race.   
  
  
**Figure B1**  
*Univariate distributions of the RIASEC subscale scores ($N = 55,593$)*
![](Paper_files/figure-docx/unnamed-chunk-32-1.png)<!-- -->
  
  
  
  
**Table B1**  
*Pearson correlations between RIASEC subscale scores ($N = 55,593$)*

|     |  Real|  Inve|   Arti|  Soci|  Ente|   Conv|
|:----|-----:|-----:|------:|-----:|-----:|------:|
|Real | 1.000| 0.332|  0.182| 0.049| 0.305|  0.460|
|Inve | 0.332| 1.000|  0.329| 0.141| 0.016|  0.065|
|Arti | 0.182| 0.329|  1.000| 0.290| 0.253| -0.056|
|Soci | 0.049| 0.141|  0.290| 1.000| 0.356|  0.124|
|Ente | 0.305| 0.016|  0.253| 0.356| 1.000|  0.464|
|Conv | 0.460| 0.065| -0.056| 0.124| 0.464|  1.000|


\newpage
# Appendix C: Variable contributions from the item-level predictive models

In Figure C1, the variable contributions are depicted for the models fitted using the RIASEC item scores as predictors. The plots show a similar, but more nuanced view, compared to the variable contributions in the models fitted using the subscale scores (Figure 3, main paper). Again, the Social preferences scale contains the strongest predictors of university major completed, followed by the Realistic subscale, then followed by the Enterprising, Investigative and Conventional preferences scales. The item-level variable contributions do provide a more nuanced view. For example, in the analyses using the item scores as predictors, the Investigative subscale does seem to contribute more strongly compared to the analyses based on the subscales.

All methods found items S5 ("I would like to help people with family-related problems") and S3 ("I would like to help people who have problems with drugs or alcohol") and to contribute most. For the penalized GLM, the GAM and the boosted ensemble, this was followed by I2 ("I would like to study animal behavior"). For the single tree, the third most important predictor was item R6 ("I would like to fix a broken faucet"). For the prediction rule ensemble and the random forest, the third and fourth most important predictors were items R3 ("I would like to work on an offshore oil-drilling rig") and E5 ("I would like to manage a department within a large company").  


**Figure C1**  
*Variable contributions for each of the models fitted using RIASEC item scores as predictors*
![](Paper_files/figure-docx/unnamed-chunk-34-1.png)<!-- -->

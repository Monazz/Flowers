1. Ask yourself why would they have selected this problem for the challenge? What are some gotchas in this domain I should know about?
FellowshipAI is preparing for real project which in that case hands on experience is required. Also, this Type of Question can evaluate critical thinking and searching on the Internet to find the proper solution. At the end student must be capable of applying the right code to the defined dataset.
2. What is the highest level of accuracy that others have achieved with this dataset or similar problems / datasets ? Deep learning or ML is based on experience and try and attempt. So there is not a right answer to that. The answer can be find based on the previous models. 
3. What types of visualizations will help me grasp the nature of the problem / data? I have chose the computer vision problem. So in my case the best  visualizations is image. At the end, in order to compare the result I have compared the "Accuracy" and "validation loss"
4. What feature engineering might help improve the signal?
We used following to improve our result.
-first I have chose randomly 30 images of each class(as you remember some flowers only have 10 picture) and I used the 10 images for validation. I have tried to keep the data set balance, so the model would not train more on the flowers that has more images.
-Transfer learning by using Resnet50 model and also I applied Feature extraction and fine tuning. Therefore, Instead of retraining the entire ResNet-50 model, I use it as a feature extractor. This involves removing the top layers of the network  and using the output of the remaining layers as features. 
-data augmentation(including rotation, flipping, scaling, and cropping can increase the diversity of the training set and help the model generalize better to unseen data.)
5. Which modeling techniques are good at capturing the types of relationships I see in this data?
6. Now that I have a model, how can I be sure that I didn't introduce a bug in the code? 
Visualize, visualize and visualize. If results are too good to be true, they probably are!
7. What are some of the weaknesses of the model and and how can the model be improved with additional work
I have work with "tf.keras.applications.efficientnet" model and it gave me a better results. Maybe, new defined model is a better choice for transfer learning.

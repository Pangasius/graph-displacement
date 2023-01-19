This contains the trained model for direct prediction on unnormalized data, runnable on commit 3470032

model_X.pkl is the model at a particular epoch X 
Losses_X.pdf is a graphical representation of the test loss over X epochs
result_X.pkl is the data that was used to create the animation (ran through the model)
animation.mp4 is the results

Commentary on results :

The data is in a world where falling off the edge wraps you to the other side.
This creates problems since the model doesn't know where those are.
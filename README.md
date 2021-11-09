# Distance Classifier

We see many in other non-computational fields running classifiers that incorrectly classify something as a class when in reality it is "none of the above". An example of this would be given a classifier that can classify dogs and cats, but is given a car. This classifier will incorrectly classify the car as either a dog or a cat when in reality, it is "None of the above". 

Currently, there are no algorithms that can accurately classify any given point as "none of the above" with a rigorous confidence value. Neural Nets are specifically hard to get something rigoous from, if you look at the mathematics of how the final number pops out, it’s basically a weighted sum of edge weights. It would be very difficult to get a number like 1e-9 out of a process like that, and The final output numbers typically are nowhere near that small. We are developing an algorithm to be able to be able to classify accurately and to produce a rigorous P value along with that classification.



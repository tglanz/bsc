## Introduction

- What is deep compression?
- Motivation for deep compression
- The standard deep compression pipeline
  - Pruning
  - Quantization
  - Huffman Coding

> Note: We are focusing on the lottery ticket hypothesis which relies on the subject of pruning. Quantization and huffman coding will only be shallowly covered in order to provide a complete picture of deep compression.

> Note: If, during writing the work we find that there is not enought substance when focusing on pruning and the lottery ticket hypothesis alone, we can dive into quantization and huffman coding as well.

## Pruning

- What is pruning?
  - Impact on Inference vs impact on Tranining
  - Sparsity in hardware
- Definitions
- Techniques
  - Weights, Neurons, Layers
  - Structured vs Unstructured
  - One Shot vs Iterative
  - Rewind

## The lottery ticket hypothesis

- Initial hypothesis
- Adjustments for bigger networks
  - Linear mode connectivity
- Impact of the hypothesis
  - Practical
  - Theoretical
- Characteristics of a lottery ticket?

## References

> Note: All links below redirect to the Open University's library

- [Learning both Weights and Connections for Efficient Neural Networks. 2015](http://elib.openu.ac.il/login?url=https://search.ebscohost.com/login.aspx?direct=true&db=edsarx&AN=edsarx.1506.02626&site=eds-live&scope=site)

- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. 2015](http://elib.openu.ac.il/login?url=https://search.ebscohost.com/login.aspx?direct=true&db=edsarx&AN=edsarx.1510.00149&site=eds-live&scope=site)

- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. 2018](http://elib.openu.ac.il/login?url=https://search.ebscohost.com/login.aspx?direct=true&db=edsarx&AN=edsarx.1803.03635&site=eds-live&scope=site)

- [What is the State of Neural Network Pruning?. 2020](http://elib.openu.ac.il/login?url=https://search.ebscohost.com/login.aspx?direct=true&db=edsarx&AN=edsarx.2003.03033&site=eds-live&scope=site)

- [Gradient flow in sparse neural networks
and how lottery tickets win. 2020](http://elib.openu.ac.il/login?url=https://search.ebscohost.com/login.aspx?direct=true&db=edsarx&AN=edsarx.2010.03533&site=eds-live&scope=site)

- [SONIC: A Sparse Neural Network Inference Accelerator with Silicon Photonics for Energy-Efficient Deep Learning. 2021](http://elib.openu.ac.il/login?url=https://search.ebscohost.com/login.aspx?direct=true&db=edsarx&AN=edsarx.2109.04459&site=eds-live&scope=site)

- [Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask?. 2022](http://elib.openu.ac.il/login?url=https://search.ebscohost.com/login.aspx?direct=true&db=edsarx&AN=edsarx.2210.03044&site=eds-live&scope=site)

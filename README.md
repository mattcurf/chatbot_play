# llm_play 

A repo which illustrates multiple 'hello world' implementations of LLM using Microsoft Phi-2 LLM model with two different software stacks:
* ipex-llm: An example using [ipex-llm](https://github.com/intel-analytics/ipex-llm) 
* optium-openvino: An example using [Optimum + OpenVINO](https://huggingface.co/docs/optimum/en/index)

For each example, enter the folder and type the following:

Build the docker container including Intel GPU drivers and other dependencies for the example
```bash
$ ./build.sh
```

Run the example to download the phi-2 model and perform a very simple prompt 'What is AI?'
```bash
$ ./run.sh
```

In each folder, the Dockerfile contains all of the dependencies required to run each solution, and test.py implements the hello, world LLM app itself. 

Requires Ubuntu 24.04 for ARC GPU in-tree driver kernel 6.8 support + docker installation

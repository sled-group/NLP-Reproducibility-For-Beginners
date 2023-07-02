# NLP-Reproducibility-For-Beginners
Repo for the ACL 2023 Paper "[NLP Reproducibility For All: Understanding Experiences of Beginners](https://arxiv.org/abs/2305.16579)" by Shane Storks, Keunwoo Peter Yu, Ziqiao Ma, and Joyce Chai.

While our IRB approval prevents us from releasing the data collected in this work, we share the data analysis and graph generation code for repeat or related studies to refer to. 

# Getting Started
Set up the Anaconda environment: 

```
conda env create -f environment.yml
```

You'll need to populate the data file `data_files/sample_data.json` with real subject data, then you can run scripts in `scripts/data_analysis` as needed. To generate the data file, we used several steps of data preprocessing to compile the subject data from multiple sources and de-identify it. Feel free to contact us if you have any questions or would like some pointers.

# Citation
```
@inproceedings{storks2023nlp,
  title={NLP Reproducibility For All: Understanding Experiences of Beginners},
  author={Storks, Shane and Yu, Keunwoo Peter and Ma, Ziqiao and Chai, Joyce},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Long Paper)},
  address={Toronto, ON, Canada},
  year={2023}
}
```


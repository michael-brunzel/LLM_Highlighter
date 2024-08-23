## Data generation process
This folder contains all the relevant files for creating the data generation workflow which is displayed in the overall README.


#### Metadata
We use some additional metadata such as location names and actual names, which are stored in *origin.py* and *name_list.py*. Some additional metadata is stored in *build_category_samples.py*

#### Build category samples
We start in file *build_category_samples.py* with generating the different sections of the CV separately. This has the advantage of generating (temporally) coherent texts more easily and helps with increasing the diversity by sampling different sections of the same style for generating the whole CV. This is considered by adapting the prompts accordingly to avoid incoherence. Six different categories are created: Personal Information, Skills, Education, Work Experience, Academia and Hobbies.

#### Build CV
The file *build_CV.py* is the last step when it comes to the creation of the input samples. To do that we sample core facts about the profile and then use our collected metadata to filter down the number of texts that match these core facts in a coherent way.
We then combine the different categories to construct the whole CV and randomly drop some of the categories to create additional variance within the dataset

#### Labels
In a final step we need to create the label strings that correspond to the created inputs. This is done in the file *create_LLM_outputs.py*

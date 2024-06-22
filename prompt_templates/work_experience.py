work_prompt = """Create a short paragraph ({sentence_length} bullet points) about a work experience of a {position} in CV style with start-end, role and so on for a person that graduated in {grad_year}
Example:
"
Data Scientist, 2018 - 2020 at SAP
- Responsible for ...
"
"""

# work_prompt_multi = """
# Create a short paragraph ({sentence_length} bullet points) about a work experience of a {position} in CV style with start-end, role and so own
# Example:
# "
# Data Scientist, 2018 - 2020 at SAP
# - Responsible for ...
# "
# This is the existing work experience of the person. Add another work experience, which is consistent:
# Old work experience:
# "
# {existing_work_experience}
# "
# """

work_prompt_multi2 = """
Create a short paragraph ({sentence_length} bullet points) about two work experiences of a {position} in CV style with start-end, role and so on for a person that graduated in {grad_year}
Example:
"
Data Scientist, 2018 - 2020 at SAP
- Responsible for ...

Junior Data Scientist, 2016 - 2018 at Merck
- Responsible for ...
"
"""

work_prompt_sentences = """
Create {sentence_length} sentences about work experiences of a {position} with role, made up company name, start and end so on. The person graduated in {grad_year}.
"""
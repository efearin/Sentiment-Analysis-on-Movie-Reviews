__author__='efe arin'

import pandas as pd
import func_repo
import io_repo
import calculate_repo

# in func repo convert numbers to strings

data_divide_constant = 5
fake_data_divide_constant = 2
list_path = 'data/lists'

# get data
df=pd.read_csv('data/data.csv',sep="\t")
df=df.head(n=1000)
# clean data
func_repo.clean_data(df)
# divide data and get df list
clean_df_list = func_repo.divide_data(df, data_divide_constant*fake_data_divide_constant)
io_repo.save(clean_df_list,list_path+'/clean_df_list')
# get fully cleaned df list
fully_clean_df_list = func_repo.get_fully_cleaned_df_list(clean_df_list)
io_repo.save(fully_clean_df_list,list_path+'/fully_clean_df_list')
# get sentence and word lists
sentence_df_list, word_df_list = func_repo.get_sentence_word_lists(fully_clean_df_list)
io_repo.save(sentence_df_list,list_path+'/sentence_df_list')
io_repo.save(word_df_list,list_path+'/word_df_list')


for x in range(0, data_divide_constant):
    # create required df_lists
    test_df_list, train_df_list, train_sentence_df_list, train_word_df_list = func_repo.get_required_df_lists(
        clean_df_list, fully_clean_df_list, sentence_df_list, word_df_list, fake_data_divide_constant, x)
    # collect feature vectors for train set
    for y in range (0, len(train_df_list)):
        # prepare necessary df's
        fake_test_df, fake_train_df, fake_train_sentence_df, fake_train_word_df = func_repo.get_required_dfs (
            train_df_list, train_sentence_df_list, train_word_df_list, y)
        # train w2v algorithm
        fake_w2v_model = func_repo.get_w2v_model(fake_train_sentence_df)
        # normalization for feature vector parameters needed output returns already normalized
        partial_mlp_train_input, partial_mlp_train_output = calculate_repo.get_matrix(fake_test_df, fake_train_df,
                                                                      fake_train_word_df, fake_w2v_model)
        a=5




    a=5

    # train_df = pd.concat(partial_df_list, ignore_index=True)
    # train_df = train_df.reset_index(drop=True)
    # test_df = test_df.reset_index(drop=True)
    # io_repo.save(fake_w2v_model, 'data/model')
The datasets were adapted from P5 (https://github.com/jeykigung/P5/tree/main/preprocess), but they are not well documented. In this work, we removed redundant data entries and wrote some notes so as to improve the readability.

datamaps.json
{	
	'user2id': user2id,
        'item2id': item2id,
}
Both user2id and item2id are dictionaries, in which each entry is a pair of index (str) and its original ID (reviewerID for users and asin for items) just in case you might want to map them back to the original datasets.
The indices for both users and items start from 1 and accumulate to the total number of users and items.

sequential.txt
Each line in this file consists of a user and his/her interacted items (sequentially ordered) represented by their indices as explained above. The user and items are separated by white space as follows.
user item1 item2 ... \n

negative.txt
Each line in this file contains sampled negative items for the target user. The data format is the same as sequential.txt
user item1 item2 ... \n

explanation.json
{
	'train': train,
	'val': val,
	'test': test,
}
train/val/test is a list of dictionaries, each of which is formatted as follows.
{
	'user': user, # int
	'item': item, # int
	'explanation': explanation, # an explanation sentence from the user's review
	'feature': feature, # a feature in the explanation sentence
}
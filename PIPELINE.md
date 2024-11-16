```mermaid
flowchart TD
	node1["embed_articles"]
	node2["embed_users"]
	node3["format_articles"]
	node4["generate_history"]
	node5["generate_user_profiles"]
	node6["make_articles"]
	node3-->node1
	node3-->node4
	node5-->node2
	node5-->node4
	node6-->node3
```

```mermaid
flowchart TD
	node1["embed_users"]
	node2["generate_user_profiles"]
	node2-->node1
	node3["embed_articles"]
	node4["format_articles"]
	node5["generate_history"]
	node6["make_articles"]
	node4-->node3
	node4-->node5
	node6-->node4
```

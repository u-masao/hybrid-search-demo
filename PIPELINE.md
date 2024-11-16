```mermaid
flowchart TD
	node1["embed_articles"]
	node2["format_articles"]
	node3["generate_history"]
	node4["generate_user_profiles"]
	node5["make_articles"]
	node2-->node1
	node2-->node3
	node4-->node3
	node5-->node2
```

test::
	pytest -v tests

format::
	toml-sort pyproject.toml

build:: format
	poetry build

publish:: build
	poetry publish 

test_heartbeat::
	## Test on 8081
	curl -X GET http://localhost:8081/system/heartbeat

test_recommend::
	## Test with a curl with user 0, passing data
	curl -X POST http://localhost:8081/user/recommend -H "Content-Type: application/json" \
		-d '{"user_id": "0", "n": 10 }' | jq .
	
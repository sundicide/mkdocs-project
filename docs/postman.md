# postman

## Pre-request Script

```javascript
const dburl = pm.environment.get("db_url");
const uid = pm.environment.get("uid");
const url = `${dburl}/db/target`;

pm.sendRequest({
  url,
  method: "POST",
  header: {
    'Accept': 'application/json',
    'Content-Type': 'application/x-www-form-urlencoded',
    id: "postman"
  },
  body: {
    mode: "raw",
    raw: JSON.stringify({
      no_seed: false,
      seed_pactivity: 8,
      protein_id: [168834],
      uid: "TEST"
    })
  }
}, function (err, response) {
  pm.environment.set("responseId", response.id);
  console.log(response.json());
});
```

## body

- body에 env값을 넣고 싶을 경우

```json
{
  "id": "{{my_id}}",
  "message": "{{my_message}}"
}
```

## tests

- set env using response

```javascript
pm.test("Status code is 200", function () {
  pm.response.to.have.status(200);
  const response = pm.response.json();
  const data = response.data;
  
  const found = data.find((d) => d.registered === false);
  if (found) {
    pm.environment.set("my_id", found.id);
  }
});
```
- get env value

```javascript
const url = pm.environment.get("db_host") + "/target";
const envUserId = pm.environment.get("id");
const envUniprot = pm.environment.get("uid");

pm.test("Status code is 200", function () {
  pm.response.to.have.status(200);
  const result = pm.response.json();
  console.log(result);
});
```

- equal check

```javascript
pm.expect(envValue).to.eql(myName);
```

- not equal check

```javascript
pm.expect(envValue).not.eql(myName);
```

- response text indclude text

```javascript
pm.expect(pm.response.text()).to.include("id");
```

const express = require("express");
const router = express.Router();
var cors = require("cors");
const app = express();

// Here we are configuring express to use body-parser as middle-ware.
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

app.use(cors());

var data = {};

router.post("/data", (request, response) => {
  console.log("POST");
  console.log(request.body);
  data = request.body;
  response.send({ status: "ok" });
});

router.get("/peaks", (request, response) => {
  console.log("GET");
  response.send(data);
});

app.use("/", router);

app.listen(3001, () => {
  console.log("Started on PORT 3001");
});

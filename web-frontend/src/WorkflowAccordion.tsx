import Button from '@mui/material/Button';
import './App.css'

import { createApiClient } from "./api/api.ts";

const api = createApiClient({
    baseUrl: "https://6j5e012gi0.execute-api.eu-central-1.amazonaws.com",
    routePath: "/image",
})

function WorkflowAccordion() {
  const handleClick = async () => {
      const response = await api.getInputImageUrl("2019-09-17_13_33_03_497.jpg");
      console.log(response);
  }

  return (
      <>
          <Button variant="contained">Hello world</Button>
          <button onClick={() => handleClick()}>
              Click for fun
          </button>
      </>
  )
}

export default WorkflowAccordion

import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

import { createApiClient } from "./api/api.ts";

const api = createApiClient({
    baseUrl: "https://6j5e012gi0.execute-api.eu-central-1.amazonaws.com",
    routePath: "/image",
})

function App() {
  const [count, setCount] = useState(0);
  const handleClick = async () => {
      const response = await api.getInputImageUrl("2019-09-17_13_33_03_497.jpg");
      console.log(response);
  }

  return (
      <>
          <h1>Vite + React</h1>
          <div className="card">
              <button onClick={() => setCount((count) => count + 1)}>
                  count is {count}
              </button>
              <p>
                  Edit <code>src/App.tsx</code> and save to test HMR
              </p>
          </div>
          <p className="read-the-docs">
              Click on the Vite and React logos to learn more
          </p>
          <button onClick={() => handleClick()}>
              Click for fun
          </button>
      </>
  )
}

export default App

import Button from '@mui/material/Button';
import './App.css'
import {createApiClient} from "./api/api.ts";
import * as React from 'react';
import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import Typography from '@mui/material/Typography';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Box from '@mui/material/Box';


function WorkflowAccordion() {
    const api = createApiClient({
        baseUrl: "https://6j5e012gi0.execute-api.eu-central-1.amazonaws.com",
        routePath: "/image",
    });
    const [expanded, setExpanded] = React.useState(false);
    const [firstImageUrl, setFirstImageUrl] = React.useState("");

    const handleChange = (panel) => (event, isExpanded) => {
        setExpanded(isExpanded ? panel : false);
    };
    const handleFirstStep = async () => {
        const response = await api.getInputImageUrl("2019-09-17_13_33_03_497.jpg");
        //response has field url
        setFirstImageUrl(response);
        console.log(response);
    }

    return (
        <div>
            <Accordion expanded={expanded === 'panel1'} onChange={handleChange('panel1')}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon/>}
                    aria-controls="panel1bh-content"
                    id="panel1bh-header"
                >
                    <Typography component="span" sx={{width: '33%', flexShrink: 0}}>
                        Step 1
                    </Typography>
                    <Typography component="span" sx={{color: 'text.secondary'}}>
                        Gather initial dataset
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    {firstImageUrl &&
                        <Box
                            component="img"
                            src={firstImageUrl}
                            alt="First Step Image"
                            sx={{
                                width: "100%",
                                height: "auto",
                                maxWidth: 600,
                                display: 'block',
                                mx: 'auto',
                                borderRadius:2,
                            }}
                        />
                    }
                    <Typography>
                        <Box
                            sx={{
                                display: 'flex',
                                justifyContent: 'center',
                            }}
                        >
                            <Box>
                            <ul>
                                <li>Retrieve Data from TimeLapse Robot</li>
                                <li>Reduced images: 1000 instead of 50000+</li>
                            </ul>
                        </Box>
                        </Box>
                        <Button
                            color="primary"
                            variant="contained"
                            onClick={() => {
                                handleFirstStep();
                            }}
                            sx={{
                                m: 2,
                            }}
                            disabled={!!firstImageUrl}
                        >
                            Finish step 1
                        </Button>
                    </Typography>
                </AccordionDetails>
            </Accordion>
            <Accordion expanded={expanded === 'panel2'} onChange={handleChange('panel2')}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon/>}
                    aria-controls="panel2bh-content"
                    id="panel2bh-header"
                >
                    <Typography component="span" sx={{width: '33%', flexShrink: 0}}>
                        Users
                    </Typography>
                    <Typography component="span" sx={{color: 'text.secondary'}}>
                        You are currently not an owner
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        Donec placerat, lectus sed mattis semper, neque lectus feugiat lectus,
                        varius pulvinar diam eros in elit. Pellentesque convallis laoreet
                        laoreet.
                    </Typography>
                </AccordionDetails>
            </Accordion>
            <Accordion expanded={expanded === 'panel3'} onChange={handleChange('panel3')}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon/>}
                    aria-controls="panel3bh-content"
                    id="panel3bh-header"
                >
                    <Typography component="span" sx={{width: '33%', flexShrink: 0}}>
                        Advanced settings
                    </Typography>
                    <Typography component="span" sx={{color: 'text.secondary'}}>
                        Filtering has been entirely disabled for whole web server
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        Nunc vitae orci ultricies, auctor nunc in, volutpat nisl. Integer sit
                        amet egestas eros, vitae egestas augue. Duis vel est augue.
                    </Typography>
                </AccordionDetails>
            </Accordion>
            <Accordion expanded={expanded === 'panel4'} onChange={handleChange('panel4')}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon/>}
                    aria-controls="panel4bh-content"
                    id="panel4bh-header"
                >
                    <Typography component="span" sx={{width: '33%', flexShrink: 0}}>
                        Personal data
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        Nunc vitae orci ultricies, auctor nunc in, volutpat nisl. Integer sit
                        amet egestas eros, vitae egestas augue. Duis vel est augue.
                    </Typography>
                </AccordionDetails>
            </Accordion>
        </div>
    )
}

export default WorkflowAccordion

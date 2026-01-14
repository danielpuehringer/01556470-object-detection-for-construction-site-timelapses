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
import JSONScreenshot from './assets/appl-dl-json.png';


function WorkflowAccordion() {
    const api = createApiClient({
        baseUrl: "https://6j5e012gi0.execute-api.eu-central-1.amazonaws.com",
        routePath: "/image",
    });
    const [expanded, setExpanded] = React.useState(false);
    const [firstImageUrl, setFirstImageUrl] = React.useState("");
    const [secondImageUrl, setSecondImageUrl] = React.useState("");

    const handleChange = (panel) => (event, isExpanded) => {
        setExpanded(isExpanded ? panel : false);
    };
    const handleFirstStep = async () => {
        const response = await api.getInputImageUrl("2019-09-17_13_33_03_497.jpg");
        setFirstImageUrl(response);
        console.log(response);
    }

    const handleSecondStep = async () => {
        const response = await api.getOutputImageUrl("2019-09-17_13_33_03_497.jpg");
        setSecondImageUrl(response);
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
                        Bring your own data - Gather initial dataset
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
                        Step 2
                    </Typography>
                    <Typography component="span" sx={{color: 'text.secondary'}}>
                        Running external object detection model (MMDet)
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        {secondImageUrl &&
                            <>
                                <Box
                                    component="img"
                                    src={secondImageUrl}
                                    alt="First Step Image"
                                    sx={{
                                        width: "100%",
                                        height: "auto",
                                        maxWidth: 600,
                                        display: 'block',
                                        mx: 'auto',
                                        borderRadius: 2,
                                    }}
                                />
                                <Box
                                    sx={{
                                            textAlign: 'left',
                                            justifyContent: 'start',
                                        }}
                                >
                                    <pre>
                                  {`{
                                    "labels": [0, 0, 0, 7, 13, ...],
                                    "scores": [
                                      0.7898109555244446,
                                      0.6911609768867493,
                                      0.649942934513092,
                                      0.5002145171165466,
                                      0.4103165566921234,
                                      ...
                                    ],
                                    "bboxes": [
                                      [428.451843261, 538.679199, 489.352966308, 662.982123],[...],...
                                    ]
                                  }`}
                                </pre>
                                </Box>
                            </>
                        }
                    </Typography>
                    <Button
                        color="primary"
                        variant="contained"
                        onClick={() => {
                            handleSecondStep();
                        }}
                        sx={{
                            m: 2,
                        }}
                        disabled={!!secondImageUrl}
                    >
                        Finish step 2
                    </Button>
                </AccordionDetails>
            </Accordion>
            <Accordion expanded={expanded === 'panel3'} onChange={handleChange('panel3')}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon/>}
                    aria-controls="panel3bh-content"
                    id="panel3bh-header"
                >
                    <Typography component="span" sx={{width: '33%', flexShrink: 0}}>
                        Step 3
                    </Typography>
                    <Typography component="span" sx={{color: 'text.secondary'}}>
                        Analyze output from MMDet
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        Understand how this can be useful...
                        <Box
                                    component="img"
                                    src={JSONScreenshot}
                                    alt="First Step Image"
                                    sx={{
                                        width: "100%",
                                        height: "auto",
                                        maxWidth: 600,
                                        display: 'block',
                                        mx: 'auto',
                                        borderRadius: 2,
                                    }}
                                />

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
                        Step 4
                    </Typography>
                    <Typography component="span" sx={{color: 'text.secondary'}}>
                        Define decision criteria for interesting images
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        <Box
                            sx={{
                                display: 'flex',
                                justifyContent: 'center',
                            }}
                        >
                            <Box>
                                <h3>What makes an interesting image?</h3>
                                <ul>
                                    <li>At least 2 people on image OR</li>
                                    <li>At least 1 person AND</li>
                                    <li>Good weather conditions</li>
                                </ul>
                            </Box>
                        </Box>
                    </Typography>
                </AccordionDetails>
            </Accordion>


            <Accordion expanded={expanded === 'panel5'} onChange={handleChange('panel5')}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon/>}
                    aria-controls="panel5bh-content"
                    id="panel5bh-header"
                >
                    <Typography component="span" sx={{width: '33%', flexShrink: 0}}>
                        Step 5
                    </Typography>
                    <Typography component="span" sx={{color: 'text.secondary'}}>
                        Generate label.csv
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        label.csv is generated based on decision criteria from Step 4 and preds file from step 2.
                        This label.csv is the base for future ML model training
                    </Typography>
                </AccordionDetails>
            </Accordion>
            
            <Accordion expanded={expanded === 'panel6'} onChange={handleChange('panel6')}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon/>}
                    aria-controls="panel6bh-content"
                    id="panel6bh-header"
                >
                    <Typography component="span" sx={{width: '33%', flexShrink: 0, fontWeight: 'bold'}}>
                        Step 6
                    </Typography>
                    <Typography component="span" sx={{color: 'text.secondary', fontWeight: 'bold'}}>
                        Label data manually (core of project!)
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        <Box
                            sx={{
                                display: 'flex',
                                justifyContent: 'center',
                                flexDirection: 'column',
                            }}
                        >
                            <Box>
                                <h3>Previous to this step I already had:</h3>
                                <ul>
                                    <li>1000 raw images</li>
                                    <li>1000 images with object detection</li>
                                    <li>A label.csv file with necessary criteria (number of persons, vehicles,...) which helped to define criteria for interesting images</li>
                                    <li>The label.csv file prefilled the "interesting" column for all images as 0 (i.e. not interesting)</li>
                                </ul>
                            </Box>
                            <Box>
                                <h3>Goal of this task</h3>
                                <ul>
                                    <li>Manually label all 1000 images and consider shortcuts</li>
                                    <li>Potential shortcut: Label all images with at least 2 persons as interesting</li>
                                    <li>Potential shortcut: Only look at images with at least 1 person and check weather condition</li>
                                    <li>Potential shortcut: Look at images with at least 1 vehicle</li>
                                    <li>Potential shortcut: Label all other images as "not interesting"</li>
                                </ul>
                            </Box>
                            <Box>
                                <h3>Result of this task</h3>
                                <ul>
                                    <li>Updated version of label.csv</li>
                                    <li>label.csv can be used for further processing</li>
                                </ul>
                            </Box>
                        </Box>
                    </Typography>
                </AccordionDetails>
            </Accordion>
            
            <Accordion expanded={expanded === 'panel7'} onChange={handleChange('panel7')}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon/>}
                    aria-controls="panel7bh-content"
                    id="panel7bh-header"
                >
                    <Typography component="span" sx={{width: '33%', flexShrink: 0}}>
                        Step 7
                    </Typography>
                    <Typography component="span" sx={{color: 'text.secondary'}}>
                        Find suitable ML model and train it
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        <Box
                            sx={{
                                display: 'flex',
                                justifyContent: 'center',
                                flexDirection: 'column',
                            }}
                        >
                            <Box>
                                <h3>Key aspects of this task</h3>
                                <ul>
                                    <li>Research which approach will be better: train model from scratch or use pre trained model</li>
                                    <li>Asked professor for insights</li>
                                    <li>Decision: Pre trained ResNet-50</li>
                                    <li>Enhancement: Researched how to optimize it (freeze backbone, etc.)</li>
                                </ul>
                            </Box>
                        </Box>
                    </Typography>
                </AccordionDetails>
            </Accordion>

            <Accordion expanded={expanded === 'panel8'} onChange={handleChange('panel8')}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon/>}
                    aria-controls="panel8bh-content"
                    id="panel8bh-header"
                >
                    <Typography component="span" sx={{width: '33%', flexShrink: 0}}>
                        Step 8
                    </Typography>
                    <Typography component="span" sx={{color: 'text.secondary'}}>
                        Run model on new data
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        <Box
                            sx={{
                                display: 'flex',
                                justifyContent: 'center',
                                flexDirection: 'column',
                            }}
                        >
                            <Box>
                                <h3>Results of model:</h3>
                                <pre
                                    sx={{
                                        width: '100%',
                                        maxWidth: "100px",
                                        textAlign: 'left',
                                        marginLeft: '0',
                                    }}
                                >
{`
Epoch 01/30 | Training: train-loss 0.3571, train-accuracy 0.855 train-precision 0.738, train-recall 0.403, train-F1 0.521
| Validation: val-loss 0.1474, val-accuracy 0.944 val-precision 0.850 val-recall 0.872 val-F1 0.861
Epoch 10/30 | Training: train-loss 0.0688, train-accuracy 0.972 train-precision 0.902, train-recall 0.961, train-F1 0.931
| Validation: val-loss 0.1900, val-accuracy 0.934 val-precision 0.861 val-recall 0.795 val-F1 0.827
Epoch 30/30 | Training: train-loss 0.0798, train-accuracy 0.967 train-precision 0.895, train-recall 0.942, train-F1 0.918
| Validation: val-loss 0.2196, val-accuracy 0.954 val-precision 0.857 val-recall 0.923 val-F1 0.889
`}
                                </pre>
                            </Box>
                        </Box>
                    </Typography>
                </AccordionDetails>
            </Accordion>
        </div>
    )
}

export default WorkflowAccordion

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Container,
  TextField,
  Button,
  Typography,
  Paper,
  CircularProgress,
  IconButton,
  Tooltip,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  useTheme,
  ThemeProvider,
  createTheme,
} from '@mui/material';
import {
  Send as SendIcon,
  Menu as MenuIcon,
  Science as ScienceIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import axios from 'axios';
import * as d3 from 'd3';

// Create a dark theme with quantum-inspired colors
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00bcd4', // Quantum blue
    },
    secondary: {
      main: '#ff4081', // Quantum pink
    },
    background: {
      default: '#0a1929',
      paper: '#1a2027',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'quantum';
  timestamp: Date;
  confidence?: number;
  quantumState?: {
    amplitudes: number[];
    phases: number[];
  };
}

const QuantumStateVisualization: React.FC<{ quantumState: Message['quantumState'] }> = ({ quantumState }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!quantumState || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 200;
    const height = 200;
    const radius = Math.min(width, height) / 2;

    const g = svg
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    // Create quantum state visualization
    const data = quantumState.amplitudes.map((amp, i) => ({
      amplitude: amp,
      phase: quantumState.phases[i],
      angle: (i * 2 * Math.PI) / quantumState.amplitudes.length,
    }));

    // Draw amplitude circles
    data.forEach((d) => {
      const x = radius * d.amplitude * Math.cos(d.angle);
      const y = radius * d.amplitude * Math.sin(d.angle);

      g.append('circle')
        .attr('cx', x)
        .attr('cy', y)
        .attr('r', 4)
        .attr('fill', `hsl(${(d.phase * 360) / (2 * Math.PI)}, 70%, 50%)`);
    });

    // Draw phase lines
    data.forEach((d) => {
      const x = radius * d.amplitude * Math.cos(d.angle);
      const y = radius * d.amplitude * Math.sin(d.angle);

      g.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', x)
        .attr('y2', y)
        .attr('stroke', `hsl(${(d.phase * 360) / (2 * Math.PI)}, 70%, 50%)`)
        .attr('stroke-width', 1)
        .attr('opacity', 0.5);
    });

    // Draw outer circle
    g.append('circle')
      .attr('r', radius)
      .attr('fill', 'none')
      .attr('stroke', '#00bcd4')
      .attr('stroke-width', 1)
      .attr('opacity', 0.3);
  }, [quantumState]);

  return (
    <svg
      ref={svgRef}
      width={200}
      height={200}
      style={{ margin: '10px auto', display: 'block' }}
    />
  );
};

const ChatMessage: React.FC<{ message: Message }> = ({ message }) => {
  const theme = useTheme();

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: message.sender === 'user' ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
    >
      <Paper
        elevation={3}
        sx={{
          p: 2,
          maxWidth: '70%',
          backgroundColor:
            message.sender === 'user'
              ? theme.palette.primary.dark
              : theme.palette.background.paper,
          borderRadius: 2,
        }}
      >
        <Typography variant="body1">{message.text}</Typography>
        {message.confidence && (
          <Typography
            variant="caption"
            sx={{ display: 'block', mt: 1, opacity: 0.7 }}
          >
            Confidence: {(message.confidence * 100).toFixed(1)}%
          </Typography>
        )}
      </Paper>
      {message.quantumState && (
        <Box sx={{ mt: 1 }}>
          <QuantumStateVisualization quantumState={message.quantumState} />
        </Box>
      )}
    </Box>
  );
};

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        message: input,
      });

      const quantumMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response.data.response,
        sender: 'quantum',
        timestamp: new Date(),
        confidence: response.data.confidence,
        quantumState: response.data.quantum_state,
      };

      setMessages((prev) => [...prev, quantumMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      // Handle error appropriately
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
        <Drawer
          anchor="left"
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
        >
          <List sx={{ width: 250 }}>
            <ListItem button>
              <ListItemIcon>
                <ScienceIcon />
              </ListItemIcon>
              <ListItemText primary="Quantum Settings" />
            </ListItem>
            <ListItem button>
              <ListItemIcon>
                <SettingsIcon />
              </ListItemIcon>
              <ListItemText primary="Preferences" />
            </ListItem>
            <ListItem button>
              <ListItemIcon>
                <InfoIcon />
              </ListItemIcon>
              <ListItemText primary="About" />
            </ListItem>
          </List>
        </Drawer>

        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <Box
            sx={{
              p: 2,
              borderBottom: 1,
              borderColor: 'divider',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <IconButton
              edge="start"
              color="inherit"
              onClick={() => setDrawerOpen(true)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" component="div">
              Quantum Chat
            </Typography>
          </Box>

          <Container maxWidth="md" sx={{ flexGrow: 1, py: 2, overflow: 'auto' }}>
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </Container>

          <Box
            sx={{
              p: 2,
              borderTop: 1,
              borderColor: 'divider',
              bgcolor: 'background.paper',
            }}
          >
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                disabled={loading}
              />
              <Button
                variant="contained"
                color="primary"
                onClick={handleSend}
                disabled={loading}
                endIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
              >
                Send
              </Button>
            </Box>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default App; 
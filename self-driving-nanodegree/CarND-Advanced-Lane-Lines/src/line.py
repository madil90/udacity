from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    def add_to_queue(self, m_queue, m_element):
        if (len(m_queue) >= self.n):
            m_queue.popleft()
        m_queue.append(m_element)

    def __init__(self):
        # maximum n
        self.n = 5
        # list for coefficients
        self.coeffs_x = deque(maxlen=self.n)
        self.coeffs_y = [None]* self.n
        # list of x, y points
        self.points = [None]*self.n
        # current best fit, x, y points 
        self.current_coeffs = [None]*self.n
        self.current_points = [None]*self.n


    def add_iteration(self, fit_x, fit_y):
        # add to the end of a list 
        self.add_to_queue(self.coeffs_x, fit_x)
        self.add_to_queue(self.coeffs_y, fit_y)

        # also add the actual points (should be final ones)
        
        
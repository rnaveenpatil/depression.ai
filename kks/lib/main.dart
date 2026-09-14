import 'package:flutter/material.dart';

void main() => runApp(const KidsApp());

class KidsApp extends StatelessWidget {
  const KidsApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'KKS',
      theme: ThemeData(
        primarySwatch: Colors.orange,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.orange),
      ),
      home: const KidsHomePage(),
    );
  }
}

class KidsHomePage extends StatelessWidget {
  const KidsHomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Kids KKS'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            const Text(
              'Welcome to Kids KKS!',
              style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 24),
            Expanded(
              child: GridView.count(
                crossAxisCount: 2,
                mainAxisSpacing: 16,
                crossAxisSpacing: 16,
                children: [
                  _buildKidButton(context, 'Play', Icons.play_arrow, Colors.green),
                  _buildKidButton(context, 'Learn', Icons.book, Colors.blue),
                  _buildKidButton(context, 'Games', Icons.gamepad, Colors.red),
                  _buildKidButton(context, 'Music', Icons.music_note, Colors.purple),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildKidButton(BuildContext context, String label, IconData icon, Color color) {
    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        primary: color,
        padding: const EdgeInsets.symmetric(vertical: 24),
      ),
      onPressed: () {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('$label pressed')),
        );
      },
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 48, color: Colors.white),
          const SizedBox(height: 8),
          Text(label, style: const TextStyle(fontSize: 20, color: Colors.white)),
        ],
      ),
    );
  }
}

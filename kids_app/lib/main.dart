import 'package:flutter/material.dart';

void main() => runApp(const KidsApp());

class KidsApp extends StatelessWidget {
  const KidsApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Kids App',
      theme: ThemeData(
        primarySwatch: Colors.orange,
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
        title: const Text('Welcome to Kids App'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: GridView.count(
          crossAxisCount: 2,
          children: [
            _buildCard(context, 'Games', Icons.games, Colors.red),
            _buildCard(context, 'Stories', Icons.book, Colors.blue),
            _buildCard(context, 'Music', Icons.music_note, Colors.green),
            _buildCard(context, 'Learn', Icons.school, Colors.purple),
          ],
        ),
      ),
    );
  }

  Widget _buildCard(BuildContext context, String title, IconData icon, Color color) {
    return Card(
      color: color.withOpacity(0.7),
      child: InkWell(
        onTap: () {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('$title tapped')),
          );
        },
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 48, color: Colors.white),
              const SizedBox(height: 8),
              Text(title, style: const TextStyle(color: Colors.white, fontSize: 18)),
            ],
          ),
        ),
      ),
    );
  }
}
